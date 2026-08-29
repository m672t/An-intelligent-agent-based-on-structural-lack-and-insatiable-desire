import matplotlib.pyplot as plt
import numpy as np

# داده‌های فرضی (بعد از اجرا، داده‌های واقعی رو جایگزین کن)
episodes = np.arange(1, 21)

# اینا رو بعد از اجرای واقعی با داده‌های خودت پر کن
baseline_rewards = np.random.normal(50, 10, 20)  # موقت
lack_rewards = np.random.normal(70, 15, 20)     # موقت

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(episodes, baseline_rewards, 'b-', label='عامل پایه')
plt.plot(episodes, lack_rewards, 'r-', label='عامل با فقدان')
plt.xlabel('Episode')
plt.ylabel('پاداش کل')
plt.title('مقایسه عملکرد')
plt.legend()
plt.grid(True)

# نمودار تنوع رفتار (موقت)
plt.subplot(1, 2, 2)
entropy_baseline = np.random.uniform(0.5, 1.0, 20)
entropy_lack = np.random.uniform(1.0, 1.8, 20)
plt.plot(episodes, entropy_baseline, 'b-', label='عامل پایه')
plt.plot(episodes, entropy_lack, 'r-', label='عامل با فقدان')
plt.xlabel('Episode')
plt.ylabel('آنتروپی سیاست')
plt.title('مقایسه تنوع رفتار')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('comparison_plots.png', dpi=300)
plt.show()
print("📊 نمودارها ذخیره شدند!")
