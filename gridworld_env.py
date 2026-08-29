import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class DynamicGridWorldEnv(gym.Env):
    """
    محیط GridWorld پویا با منابع متحرک
    - عامل باید به سمت منابع حرکت کند
    - منابع هر چند قدم یکبار جابه‌جا می‌شوند
    - حداکثر پاداش نظری وجود دارد اما دست‌یافتنی نیست
    """
    
    def __init__(self, size=8, max_steps=100, max_theoretical_reward=1000):
        super(DynamicGridWorldEnv, self).__init__()
        
        self.size = size
        self.max_steps = max_steps
        self.max_theoretical_reward = max_theoretical_reward
        self.current_step = 0
        
        # فضای اقدامات: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # فضای مشاهدات: موقعیت عامل + موقعیت منبع
        self.observation_space = spaces.Box(
            low=0, high=size-1, shape=(4,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # موقعیت شروع تصادفی
        self.agent_pos = np.array([
            random.randint(0, self.size-1),
            random.randint(0, self.size-1)
        ])
        
        # موقعیت منبع (پاداش)
        self.resource_pos = self._generate_resource_position()
        
        self.current_step = 0
        self.total_reward = 0
        
        return self._get_obs(), {}
    
    def _generate_resource_position(self):
        """تولید موقعیت تصادفی برای منبع"""
        while True:
            pos = np.array([
                random.randint(0, self.size-1),
                random.randint(0, self.size-1)
            ])
            # منبع نباید روی عامل باشد
            if not np.array_equal(pos, self.agent_pos):
                return pos
    
    def _get_obs(self):
        """بازگشت مشاهدات: موقعیت عامل + موقعیت منبع"""
        return np.concatenate([self.agent_pos, self.resource_pos]).astype(np.float32)
    
    def _move_resource(self):
        """جابه‌جایی تصادفی منبع (شبیه‌سازی محیط پویا)"""
        # با احتمال 20% منبع جابه‌جا می‌شود
        if random.random() < 0.2:
            self.resource_pos = self._generate_resource_position()
    
    def step(self, action):
        self.current_step += 1
        
        # اجرای اقدام
        if action == 0:  # بالا
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # پایین
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # چپ
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # راست
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        
        # جابه‌جایی منبع (محیط پویا)
        self._move_resource()
        
        # محاسبه پاداش
        distance = np.linalg.norm(self.agent_pos - self.resource_pos)
        
        # پاداش خارجی (وظیفه): هرچه نزدیک‌تر، پاداش بیشتر
        reward_task = max(0, 10 - distance)  # حداکثر 10
        if distance == 0:
            reward_task = 20  # پاداش ویژه برای رسیدن به منبع
        
        # اگر به منبع رسید، منبع جدید ایجاد کن
        if distance == 0:
            self.resource_pos = self._generate_resource_position()
        
        # بررسی پایان
        terminated = (self.current_step >= self.max_steps)
        truncated = False
        
        self.total_reward += reward_task
        
        return self._get_obs(), reward_task, terminated, truncated, {
            'distance': distance,
            'total_reward': self.total_reward
        }
    
    def render(self, mode='human'):
        """نمایش بصری محیط"""
        grid = np.full((self.size, self.size), '.', dtype=str)
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'  # عامل
        grid[self.resource_pos[0], self.resource_pos[1]] = 'R'  # منبع
        
        print('-' * (self.size * 2 + 1))
        for row in grid:
            print('|' + '|'.join(row) + '|')
        print('-' * (self.size * 2 + 1))
        print(f"Step: {self.current_step}/{self.max_steps}, Distance: {np.linalg.norm(self.agent_pos - self.resource_pos):.2f}")
