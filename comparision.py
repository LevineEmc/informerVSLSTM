import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子（保证每次图一样）
np.random.seed(42)

# 模拟真实时间序列数据
time_steps = 200
x = np.arange(time_steps)
true_values = np.sin(0.05 * x) + np.random.normal(0, 0.05, time_steps)

# 模拟LSTM预测（误差稍大）
lstm_pred = true_values + np.random.normal(0, 0.15, time_steps)

# 模拟Informer预测（误差更小）
informer_pred = true_values + np.random.normal(0, 0.08, time_steps)

# 绘图
plt.figure(figsize=(10,5))

plt.plot(true_values, label='True Values', linewidth=2)
plt.plot(lstm_pred, label='LSTM Prediction', linestyle='--')
plt.plot(informer_pred, label='Informer Prediction', linestyle=':')

plt.title("Prediction Comparison: True vs LSTM vs Informer")
plt.xlabel("Time Step")
plt.ylabel("Value")

plt.legend()
plt.grid(True)

plt.tight_layout()

# 保存图片
plt.savefig("prediction_comparison.png", dpi=300)

plt.show()