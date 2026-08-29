from gridworld_env import DynamicGridWorldEnv
import numpy as np

env = DynamicGridWorldEnv(size=6, max_steps=30)

obs, info = env.reset()
print("موقعیت عامل:", obs[0:2])
print("موقعیت منبع:", obs[2:4])

for step in range(10):
    action = np.random.randint(0, 4)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(f"قدم {step+1}: پاداش={reward}")
    if terminated:
        break
