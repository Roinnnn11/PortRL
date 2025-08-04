import torch
import torch.optim as optim
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gym
from Agent.Encoder import ActorCritic
from torch.distributions import Categorical
from previous.env import DockerYard
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_state(state_np):
    """
    将 numpy 格式的 state 拆分成 current 和 history，用于模型输入。
    """
    # state_np shape: (961, 20)
    state_tensor = torch.tensor(state_np, dtype=torch.float32)

    # 假设第一个箱子是当前箱子，剩下的是历史（你可以根据优先级或策略选）
    current = state_tensor[0].unsqueeze(0).unsqueeze(0)   # [1, 1, 20]
    history = state_tensor[1:].unsqueeze(0)               # [1, 960, 20]

    return current, history


MAX_HISTORY = 5
class PPO:
    def __init__(self, input_dim, action_dim, model_dim=128, device='cuda'):
        self.device = device

        # 初始化 Actor-Critic 网络
        self.model = ActorCritic(input_dim, model_dim, action_dim).to(device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        # 历史序列
        self.history = []
        # 其他 PPO 超参数
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.gamma = 0.99
        self.lam = 0.95

    def select_action(self, state_np, env):
        print(">>>select_action called")
        current, history = preprocess_state(state_np)
        current = current.to(self.device)
        history = history.to(self.device)

        logits, value = self.model(current, history)
        # 可能待删，第一维为batch
        logits = logits.squeeze(0)  # logits: [481]

        # 得到合法动作列表
        valid_actions = env.unwrapped.get_valid_actions()
        valid_indices = [i for i, v in enumerate(valid_actions) if v]
        # print("index",valid_indices)

        # 从合法动作中提取 logits
        #debug
        # print("action:", action)  # 或 log_prob、value
        # print("logit.shape:",logits.shape)
        # print("index:", valid_indices)  # 尤其是用来 indexing 的值
        
        if len(valid_indices) == 0:
            raise ValueError("No valid actions available!")
        
        masked_logits = logits[valid_indices]
        probs = Categorical(logits=masked_logits)
        selected_index = probs.sample().item()

        # 映射回原始动作编号
        action = valid_indices[selected_index]
        log_prob = probs.log_prob(torch.tensor(selected_index, device=self.device))

        return action, log_prob, value
    # def select_action(self, state_np):
    #     current, history = preprocess_state(state_np)  # 拆出当前箱子和其他箱子

    #     current = current.to(self.device)
    #     history = history.to(self.device)

    #     logits, value = self.model(current, history)
    #     probs = Categorical(logits=logits)
    #     action = probs.sample()


    #     print(f"[Select Action] selected action: {action.item()}")

    #     return action.item(), probs.log_prob(action), value


    # def select_action(self, current_state):
    #     # 添加当前状态到历史中
    #     if len(self.history) == 0:
    #         history_tensor = torch.zeros((1, 1, current_state.shape[-1]))  # 初始 history 是全 0
    #     else:
    #         history_tensor = torch.stack(self.history, dim=1)  # (batch=1, seq_len, dim)

    #     state = torch.tensor(current_state, dtype=torch.float32).to(self.device)
    #     history_tensor = history_tensor.to(self.device)
    #     logits, value = self.model(state,history_tensor)
    #     probs = Categorical(logits=logits)
    #     action = probs.sample()
    #     # 添加当前状态到历史中（detach 避免计算图累积）
    #     if len(self.history) >= MAX_HISTORY:
    #         self.history.pop(0)
    #     self.history.append(history_tensor.squeeze(0).detach())
    #     return action.item(), probs.log_prob(action), value

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return torch.tensor(returns).to(self.device)

    def update(self, states, actions, log_probs, returns, values):
        # 拆 batch：states 是原始 state_np
        currents = []
        histories = []
        for s in states:
            current, history = preprocess_state(s)
            currents.append(current)
            histories.append(history)

        currents = torch.cat(currents, dim=0).to(self.device)   # [B, 1, D]
        histories = torch.cat(histories, dim=0).to(self.device) # [B, T, D]
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        values = torch.stack(values).to(self.device)
        advantages = returns - values.squeeze(-1)

        for _ in range(4):  # PPO epochs
            logits, new_values = self.model(currents, histories)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = (new_log_probs - log_probs).exp()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - new_values.squeeze(-1)).pow(2).mean()
            #debug
            print(f"Value: min={values.min().item():.3f}, max={values.max().item():.3f}, mean={values.mean().item():.3f}")
            print(f"Return: min={returns.min().item():.3f}, max={returns.max().item():.3f}, mean={returns.mean().item():.3f}")
            print(f"Advantage: min={advantages.min().item():.3f}, max={advantages.max().item():.3f}, mean={advantages.mean().item():.3f}")
            # debug
            # 若 returns 是 list，则建议先转换为 tensor 后 detach
            if isinstance(returns, list):
                returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            else:
                returns = returns.detach().float().to(self.device)

            advantages = advantages.detach()
            log_probs = log_probs.detach()
            values = values.detach()
            
            loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            print(f"Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}, Entropy: {entropy.item()}")
            print(f"Total Loss: {loss.item()}"
                  "完成反向传播")
            self.optimizer.step()


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# env = gym.make("DockerYard-v0")

# model = ActorCritic(input_dim=10, model_dim=64, action_dim=env.action_space.n).to(device)
# ppo = PPO(model, device=device)

# env = DockerYard(train_type=4, with_predict=True, algorithm='ppo')
# obs = env.reset()
# done = False

# while not done:
#     obs_tensor = transform_obs(obs)  # 将你的obs["current"] / obs["history"]变成 tensor
#     logits, value = model(obs_tensor)
#     dist = Categorical(logits=logits)
#     action = dist.sample()
#     obs, reward, done, info = env.step(action.item())
#     print(f"Action: {action.item()}, Reward: {reward}, Done: {done}")

# for episode in range(1000):
#     obs = env.reset()
    
#     obs_hist = []
#     obs_cur = []
#     actions = []
#     rewards = []
#     dones = []
#     log_probs = []
#     values = []

#     done = False
#     while not done:
#         current = torch.tensor(obs["current"], dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, D)
#         history = torch.tensor(obs["history"], dtype=torch.float32).unsqueeze(0).to(device)  # (1, N_hist, D)

#         logits, value = model(current, history)
#         dist = Categorical(logits=logits)
#         action = dist.sample()

#         next_obs, reward, done, _ = env.step(action.item())

#         obs_cur.append(current)
#         obs_hist.append(history)
#         actions.append(action)
#         rewards.append(torch.tensor([reward], dtype=torch.float32, device=device))
#         dones.append(torch.tensor([done], dtype=torch.float32, device=device))
#         log_probs.append(dist.log_prob(action))
#         values.append(value.squeeze())

#         obs = next_obs

#     # Bootstrap value for the final step
#     with torch.no_grad():
#         next_value = torch.zeros(1).to(device) if done else model(
#             torch.tensor(obs["current"], dtype=torch.float32).unsqueeze(0).to(device),
#             torch.tensor(obs["history"], dtype=torch.float32).unsqueeze(0).to(device)
#         )[1].squeeze()

#     returns = ppo.compute_returns(rewards, dones, values, next_value)
#     advantages = returns - torch.stack(values).to(device)

#     # 打包 batch
#     batch = (
#         torch.cat(obs_cur),
#         torch.cat(obs_hist),
#         torch.stack(actions),
#         torch.stack(log_probs),
#         returns.detach(),
#         advantages.detach()
#     )

#     ppo.update(batch)

#     print(f"Episode {episode}, total reward: {sum([r.item() for r in rewards]):.2f}")