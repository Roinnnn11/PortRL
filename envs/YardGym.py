import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.yard_env import YardEnv

class YardGymEnv(gym.Env):
    """
    Gym-compatible wrapper for your custom yard environment.
    Assumes you already have an environment class like `YourCustomYardEnv`.
    """

    def __init__(self, options):
        super(YardGymEnv, self).__init__()

        # === 1. 初始化你已有的环境逻辑 ===
        self.yard = YardEnv(options)

        # === 2. 确定 observation 和 action space ===
        self.obs_dim = self.yard.get_observation_dim()  # 例如：状态向量维度
        self.max_actions = self.yard.MAX_ACTION_CANDIDATES  # 候选动作上限

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.max_actions)

    def reset(self,seed=123):
        """重置环境，返回初始 observation 向量"""
        self.yard.reset()  # 你自己的 reset
        return self._get_observation_vector(),{}

    def step(self, action_idx):
        """执行动作索引，返回 observation, reward, done, info"""
        obs_list, reward, done, info = self.yard.step(action_idx)
        truncated = False
        return self._get_observation_vector(obs_list), reward, done, truncated,info

    def _get_observation_vector(self, obs_list=None):
        """
        将你环境返回的 observation 列表（包含堆场、任务等）扁平化成 1D 向量
        """
        if obs_list is None:
            obs_list = self.yard._get_observation()
        flat = []
        for item in obs_list:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return np.array(flat, dtype=np.float32)

    def render(self, mode="human"):
        """可选：打印当前 yard 状态"""
        self.yard.render()  # 如果你有的话

    def close(self):
        pass
