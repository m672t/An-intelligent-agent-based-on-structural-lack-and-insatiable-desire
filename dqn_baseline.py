from gridworld_env import DynamicGridWorldEnv
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

# ایجاد محیط
env = DynamicGridWorldEnv(size=6, max_steps=50)

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
print("\n🚀 شروع آموزش عامل پایه...")
model.learn(total_timesteps=15000, log_interval=200)
print("✅ آموزش تمام شد!")

# ذخیره مدل
model.save("dqn_baseline")

# ارزیابی
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"\n📊 میانگین پاداش: {mean_reward:.2f} +/- {std_reward:.2f}")
