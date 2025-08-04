import gym
import numpy as np
from gym.spaces import Discrete, Box
from collections import deque


class YardEnv(gym.Env):
    def __init__(self, options):
        super().__init__()
        self.options = options  # 包含 yard 宽高深、最大堆位数、候选动作数等配置
        
        # 配置常量
        self.CONTAINER_FEATURE_DIM = 8 #集装箱特征维度
        self.GLOBAL_FEATURE_DIM = 4 #整体特征
        self.STACK_FEATURE_DIM = 6 # 栈特征
        self.MAX_STACK_NUM = 100
        self.MAX_ACTION_CANDIDATES = 50 #候选动作

        #管理存在/新建栈
        self.stack_map = {}  # 已存在栈（非空）
        self.available_slots = set()  # 可以新建栈的位置，例如 {(2,1), (4,3)}
        self.stack_id_counter = 1  # 每个新栈分配唯一 id
        self.container_list = []
        # 状态维度
        self.obs_dim = self.CONTAINER_FEATURE_DIM + self.GLOBAL_FEATURE_DIM + \
                       self.STACK_FEATURE_DIM * self.MAX_STACK_NUM
        self.observation_space = Box(low=-1, high=1e6, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = Discrete(self.MAX_ACTION_CANDIDATES)
        self.steps = 0
        # 环境状态变量
        self.reset()

    def reset(self):
        # 初始化yard/container/计划等
        self.operation_queue = deque(self._generate_operations())#形成操作队列
        self.yard = self._init_yard()
        self.current_container = self._next_container(0)
        self.current_container_map = {}
        self.steps = 0

        # 生成当前候选动作
        self.candidate_actions = self._update_candidate_actions()

        #设置available区域
        for bay in range(1, self.options.yard_width * 2 + 1):  # double width for finer bay spacing
            for row in range(1, self.options.yard_height + 1):
                self.available_slots.add((bay, row))

        return self._get_observation()
    
    def _generate_operations(self):#生成操作队列
        ops = []
        for i in range(10):
            container = self._next_container(i)
            ops.append({'ACTION': 'enter', 'CONTAINER': container})

        # 随后模拟所有箱子要离场
        for container in self.container_list:
            ops.append({'ACTION': 'leave', 'CONTAINER': container})
        return ops

    def get_observation_dim(self):
        return self.obs_dim
    # def step(self, action_idx):
    #     self.steps += 1
    #     done = False

    #     if not self.operation_queue:
    #         #如果没有操作队列，返回
    #         return self._get_observation(), 0, True, {}

    #     operation = self.operation_queue[0]
    #     container = operation['CONTAINER']
    #     action_type = operation['ACTION']

    #     # leave：处理是否被遮挡
    #     if action_type == 'leave':
    #         pile_code = container['START_PILE_PLACE']
    #         bay, row, tier = self._split_pile(pile_code) #解码
    #         stack = self._get_stack(bay, row)

    #         if not stack or stack[-1]['ID'] != container['ID']:
    #         # 如果被压住，则插入 move 动作
    #             blockers = []
    #             while stack and stack[-1]['ID'] != container['ID']:
    #                 blockers.append(stack[-1])
    #                 stack.pop()  # 模拟暂时移除，防止死循环
    #         # 恢复栈
    #             for c in reversed(blockers):
    #                 stack.append(c)

    #             for block in reversed(blockers):
    #                 self.operation_queue.appendleft({'ACTION': 'move', 'CONTAINER': block})
    #             return self._get_observation(), -1, False, {}

    #     # 当前候选动作
    #     if action_idx >= len(self.candidate_actions):
    #         return self._get_observation(), -10, True, {}

    #     pos = self.candidate_actions[action_idx]
    #     success = self._execute_operation(operation, pos) #执行当前动作

    #     if success:
    #         self.operation_queue.popleft()
    #         reward = 1
    #     else:
    #         reward = -5

    #     self._update_candidate_actions() #更新候选动作
    #     done = len(self.operation_queue) == 0 
    #     return self._get_observation(), reward, done, {}
    def step(self, action_idx):
        done = False
        reward = 0

        # === 1. 当前操作 ===
        if not self.operation_queue:
            return self._get_observation(), 0, True, {}

        current_op = self.operation_queue[0]
        container = current_op['CONTAINER']
        action_type = current_op['ACTION']

        # === 2. 获取候选动作 ===
        candidate_positions = self._get_candidate_positions(container)

        if action_idx >= len(candidate_positions):
            raise ValueError("Invalid action index!")

        target_pos = candidate_positions[action_idx]
        bay, row, tier = target_pos

        # === 3. 执行动作 ===
        if action_type == 'enter':
            success = self._place_container(container, bay, row)
            if success:
                reward = self._compute_reward(container, target_pos)
                self.operation_queue.popleft()

        elif action_type == 'move':
            success = self._place_container(container, bay, row)
            if success:
                reward = self._compute_reward(container, target_pos)
                self.operation_queue.popleft()

        elif action_type == 'leave':
            # 1. 先检查目标箱子是否在栈顶
            pos = self._get_container_position(container)
            if not pos:
                reward = -10  # 没找到
            else:
                bay, row, tier = pos
                stack = self._get_stack(bay, row)
                if stack[-1]['ID'] == container['ID']:
                    self._remove_container(container)
                    reward = self._compute_reward(container, pos)
                    self.operation_queue.popleft()
                else:
                    # 栈顶不是目标箱子 → 插入 move 操作
                    top_container = stack[-1]
                    move_op = {
                        'ACTION': 'move',
                        'CONTAINER': top_container
                    }
                    self.operation_queue.insert(0, move_op)
                    reward = -1  # 插入 move 惩罚（或 0）

        else:
            raise ValueError("Unknown action type: " + str(action_type))

        # === 4. 更新 observation 和动作队列 ===
        self._update_candidate_actions()
        obs = self._get_observation()
        done = self._check_done()
        return obs, reward, done, {}


    

    def _execute_operation(self, op, pos):#执行动作
        bay, row, tier = pos
        container = op['CONTAINER']
        action = op['ACTION']

        if action == 'enter' or action == 'move':
            stack = self._get_stack(bay, row)
            if len(stack) < self.options.yard_depth and self._is_stackable(stack, container):
                stack.append(container)
                container['START_PILE_PLACE'] = f"Block{bay:02d}{row:02d}{tier}"
                return True
            return False

        elif action == 'leave':
            stack = self._get_stack(bay, row)
            if stack and stack[-1]['ID'] == container['ID']:
                stack.pop()
                return True
            return False
    
    def render(self, mode="human"):
        #展示函数
        print(f"Step: {self.steps}, Yard has {len(self.current_container_map)} containers.")

    # 以下是功能函数
    def _get_observation(self):#状态构建
        container_feature = self._get_container_feature(self.current_container)
        global_feature = self._get_yard_global_feature()

        stack_features = []
        for ref_id, container in self.current_container_map.items():
            bay, row, tier = self._split_pile(container['START_PILE_PLACE'])
            feature = self._get_stack_feature(bay, row, tier)
            stack_features.append(feature)

        while len(stack_features) < self.MAX_STACK_NUM:
            stack_features.append([0] * self.STACK_FEATURE_DIM)
        stack_features = stack_features[:self.MAX_STACK_NUM]

        obs = np.concatenate([container_feature, global_feature, np.array(stack_features).flatten()])
        return obs.astype(np.float32)
    
    def _update_candidate_actions(self): #更新候选动作
        if not self.operation_queue:
            self.candidate_actions = []
            return

        current = self.operation_queue[0]
        action = current['ACTION']
        container = current['CONTAINER']
        self.current_container = container

        if action == 'enter' or action == 'move':
            self.candidate_actions = self._get_candidate_positions(container) #预选位置
        elif action == 'leave':
         # 找可从顶部直接取出的堆位
            pos = self._get_container_pos(container)
            if pos:
                self.candidate_actions = [pos]
            else:
                self.candidate_actions = []
    
    #更新摆放的侯选位置
    def _get_candidate_positions(self, container): 
        '''input:container
            output:可以摆放的候选位置'''
        candidates = []

     # 1. 现有栈中能堆的
        for sid, sinfo in self.stack_map.items():
            stack = sinfo['stack']
            if len(stack) >= self.options.yard_depth:
                continue
            if sinfo['size_class'] != container['SIZE']:
                continue
            if not self._is_stackable(stack, container):
                continue
            bay, row = sinfo['bay'], sinfo['row']
            tier = len(stack) + 1
            candidates.append([bay, row, tier])

        # 2. 新建栈（如果允许）
        if len(self.stack_map) < self.options.max_stack_num:
            for bay, row in self.available_slots:
                candidates.append([bay, row, 1])

        return candidates[:self.MAX_ACTION_CANDIDATES]
    
    #是否全部完成
    def _check_done(self):
        # 条件1：所有操作完成（operation_queue 空了）
        if len(self.operation_queue) == 0:
            return True

        # 条件2：达到最大步数（可选）
        if hasattr(self, 'step_count') and self.step_count >= self.options.max_episode_steps:
            return True

        return False



    #特征函数
    def _get_container_feature(self, container):
        return np.array([
            container['WEIGHT'],
            container['TYPE'],
            container['PRIORITY'],
            container['ARRIVAL'],
            container['DEADLINE'],
            0, 0, 0  # 占位
        ])

    def _get_stack_feature(self, bay, row, tier):
        stack = self._get_stack(bay, row)
        return np.array([
            bay, row, tier,
            len(stack),
            stack[-1]['WEIGHT'] if stack else 0,
            stack[-1]['TYPE'] if stack else 0
        ])

    def _get_yard_global_feature(self):
        total = self.options.yard_width * self.options.yard_height
        used = len(self.current_container_map)
        heights = [len(self._get_stack(i, j)) for i in range(1, self.options.yard_width * 2, 2) for j in range(1, self.options.yard_height + 1)]
        max_height = max(heights) if heights else 0
        avg_height = sum(heights) / len(heights) if heights else 0
        return np.array([max_height, avg_height, used, total - used])

    #辅助函数
    def _init_yard(self):
        # 初始化堆场数据结构，例如二维 dict 或 list
        return {}

    def _get_stack(self, bay, row):
        return self.yard.get((bay, row), [])

    def _split_pile(self, pile_code):
        # 例如 Block050207 → (5, 2, 7)
        return int(pile_code[5:7]), int(pile_code[7:9]), int(pile_code[9:])

    def _is_stackable(self, stack, container):#判断该stack可不可以堆放这个container
        if not stack:
            return True #空堆认为都可以堆放（）
        
        top = stack[-1]
        top_size = top['SIZE']
        new_size = container['SIZE']

         # 同尺寸或更大压更小可以
        if new_size <= top_size:
            return True

        # 小箱不能压大箱
        return False


    def _next_container(self,i):
        # 模拟新箱生成，或从预定义队列中取一个
        return {
            'ID': f'C{i}',
            'WEIGHT': np.random.randint(10, 50),
            'TYPE': np.random.randint(0, 3),
            'PRIORITY': np.random.randint(1, 10),
            'ARRIVAL': self.steps,
            'DEADLINE': self.steps + np.random.randint(5, 15),
            'SIZE':np.random.choice([20,40,45])
        }

    #放置集装箱
    def _place_container(self, container, bay,row):
        key = (bay, row)

        # 获取当前栈，或创建新栈
        stack = self.yard.setdefault(key, [])

        # 超过堆叠深度，不允许放置
        if len(stack) >= self.options.yard_depth:
            return False

        container_size = container['SIZE']

        if stack:
            # 非空栈：必须尺寸匹配
            top_size = stack[0]['SIZE']
            if top_size != container_size:
                return False
        else:
            # 空栈：准备新建
            #TODO：如果40ft，也可以接受两个20ft的堆叠
            bay_span = self._get_bay_span(container_size)  # 例如 40ft 箱子占用 2 个 bay
            occupied_positions = [(bay + offset, row) for offset in range(bay_span)]

            # 检查这些位置是否都在 available_slots 中
            if not all(pos in self.available_slots for pos in occupied_positions):
                return False  # 有部分空间已被占

            if self.options.max_stack_num and len(self.stack_map) >= self.options.max_stack_num:
                return False

            # 创建新栈
            self.stack_map[self.stack_id_counter] = {
                'bay': bay,
                'row': row,
                'stack': stack,
                'size_class': container_size,
                'occupied': occupied_positions  # 记录这个栈实际占的空间
            }
            self.stack_id_counter += 1

            # 将这些 bay-row 空间标记为已占用
            for pos in occupied_positions:
                self.available_slots.discard(pos)

            stack.append(container)
        return True

    
    #找到集装箱位置
    def _get_container_position(self, container):
        for sid, sinfo in self.stack_map.items():
            for idx, item in enumerate(sinfo['stack']):
                if item['ID'] == container['ID']:
                    return (sinfo['bay'], sinfo['row'], idx + 1)
        return None

    #计算占用空间
    def _get_bay_span(self, size):
        if size == 20:
            return 1
        elif size == 40:
            return 2
        elif size == 45:
            return 3
        else:
            raise ValueError("Unknown container size")

    def _compute_reward(self, container, pos):
        #TODO:需要完善
        # 可自定义：靠近出口 +1，重箱压轻箱 -1，目标堆高越低越好...
        return 1
