from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.YardGym import YardGymEnv  # 替换为你封装后的 gym 环境类
# from your_options_module import get_options  # 替换为返回参数选项的函数
class YardOptions:
    def __init__(self):
        self.yard_width = 6          # 横向 bay 数（注意是宽度）
        self.yard_height = 4         # 前后 row 数
        self.yard_depth = 4          # 每个堆叠最大层数
        self.block = "A"             # 区块标识
        self.with_predict = False    # 是否使用预测信息
        self.max_stack_num = 100     # 最多允许多少个栈
        self.use_fixed_action_space = False  # 是否固定动作空间
        self.max_episode_steps = 200
        self.random_seed = 42
        self.verbose = True

        self.max_action_candidates = 32  # 每步最多可选的动作数

def get_options():
    return YardOptions()


# 1. 准备参数选项（替换为你的真实配置）
options = get_options()  # options 应包含 yard_width, height, depth, 等属性

# 2. 创建环境
env = YardGymEnv(options)

# 3. 可选：检查是否符合 Gym 要求
check_env(env, warn=True)

# 4. 初始化 PPO 模型（使用 MLP 策略）
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
)

# 5. 开始训练
model.learn(total_timesteps=10)

# 6. 保存模型
model.save("ppo_yard_agent")

# 7. 测试模型运行
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break
