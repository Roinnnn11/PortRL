import gym
import numpy as np
from previous.env import DockerYard  # 你自己写的
from Agent.PPO import PPO
from stable_baselines3.common.monitor import Monitor
import torch
import gc
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.cuda.empty_cache()
gc.collect()
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

def train():
    env = DockerYard(train_type=4, with_predict=False, algorithm='ppo')
    env = Monitor(env)
    obs_dim = env.observation_space.shape[0]
    act_dim = 481

    agent = PPO(obs_dim, act_dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_episodes = 1000
    max_steps = 100

    for episode in range(max_episodes):
        state, _ = env.reset()
        print("state type:", type(state))
        if isinstance(state, dict):
            for k, v in state.items():
                print(f"{k}: shape={np.shape(v)}, dtype={type(v)}")
        else:
            print("state shape:", np.shape(state))
        agent.history = [] # 清空历史记录
        log_probs = []
        rewards = []
        values = []
        actions = []
        states = []
        dones = []

        for step in range(max_steps):
            action, log_prob, value = agent.select_action(state,env)

            next_state, reward, done, _,_ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            actions.append(action)
            states.append(state)
            dones.append(done)
   
            state = next_state

            if done:
                break

        with torch.no_grad():
            # _, next_value = agent.model(torch.tensor(state, dtype=torch.float32).to(device))
            current, history = preprocess_state(state)
            current = current.to(device)
            history = history.to(device)
            _, next_value = agent.model(current, history)
 
        returns = agent.compute_returns(rewards, dones, values, next_value.squeeze(-1))
        agent.update(states, actions, log_probs, returns, values)

        torch.cuda.empty_cache()
        gc.collect()

        print(f"Episode {episode} | Reward: {sum(rewards):.2f} | Steps: {step}")

if __name__ == '__main__':
    train()
