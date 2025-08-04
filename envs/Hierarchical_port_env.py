import gym
from gym import spaces
import numpy as np

class PortContainerEnv(gym.Env):
    def __init__(self, yard_size=(10, 6, 4), max_steps=200):
        """
        港口集装箱分配环境
        
        参数：
        yard_size: (width, height, depth) 堆场三维尺寸
        max_steps: 每个episode最大步数
        """
        super().__init__()
        
        # 环境参数
        self.yard_w, self.yard_h, self.yard_d = yard_size
        self.max_steps = max_steps
        self.current_step = 0
        
        # 动作空间定义（分层结构）
        self.action_space = spaces.Dict({
            "operation": spaces.Discrete(3),  # 0=存放 1=取出 2=移动至别处
            #对应位置坐标
            "position": spaces.MultiDiscrete([self.yard_w, self.yard_h,self.yard_d])
        })
        
        # 状态空间定义
        self.observation_space = spaces.Dict({
            "yard": spaces.Box(low=0, high=1, 
                             shape=(self.yard_w, self.yard_h, self.yard_d)),
            "crane": spaces.Box(low=0, high=max(yard_size[:2])), 
            "containers": spaces.Dict({
                "incoming": spaces.Box(low=0, high=10, shape=(5,)),
                "outgoing": spaces.Box(low=0, high=10, shape=(5,))
            })
        })
        
        # 初始化
        self.reset()