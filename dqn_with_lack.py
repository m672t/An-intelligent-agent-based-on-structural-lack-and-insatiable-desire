import gymnasium
from gridworld_env import DynamicGridWorldEnv
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from collections import deque

class LackWrapper(gymnasium.Wrapper):
    """
    Wrapper ای که پاداش درونی "فقدان" رو به محیط اضافه می‌کنه
    """
    def __init__(self, env, alpha=0.5, beta=0.2, window_size=10):
        super().__init__(env)
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.reward_history = deque(maxlen=window_size)
        self.demand_level = 0
        self.episode_rewards = []
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reward_history.clear()
        self.demand_level = 0
        self.episode_rewards = []
        return obs, info
    
    def step(self, action):
        obs, reward_task, terminated, truncated, info = self.env.step(action)
        
        # ذخیره پاداش
        self.reward_history.append(reward_task)
        
        # محاسبه سطح تقاضا (Demand_t)
        if len(self.reward_history) > 0:
            avg_reward = np.mean(self.reward_history)
            self.demand_level = avg_reward * (1 + self.beta)
        
        # محاسبه پاداش درونی "فقدان"
        if len(self.reward_history) > 0:
            intrinsic_reward = self.alpha * (self.demand_level - reward_task)
        else:
            intrinsic_reward = 0
        
        # پاداش کل
        total_reward = reward_task + intrinsic_reward
        
        # برای نمایش در گزارش
        info['reward_task'] = reward_task
        info['intrinsic_reward'] = intrinsic_reward
        info['total_reward'] = total_reward
        info['demand_level'] = self.demand_level
        
        return obs, total_reward, terminated, truncated, info

# ایجاد محیط با فقدان
base_env = DynamicGridWorldEnv(size=6, max_steps=50)
env = LackWrapper(base_env, alpha=0.3, beta=0.15)

# ایجاد عامل DQN
model = DQN(
    'MlpPolicy',
    env,
    learning_rate=0.001,
    buffer_size=50000,
    learning_starts=100,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    target_update_interval=100,
    verbose=1
)

# آموزش
print("\n🚀 شروع آموزش عامل با فقدان...")
model.learn(total_timesteps=15000, log_interval=200)
print("✅ آموزش تمام شد!")

# ذخیره مدل
model.save("dqn_with_lack")

# ارزیابی
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"\n📊 میانگین پاداش: {mean_reward:.2f} +/- {std_reward:.2f}")
