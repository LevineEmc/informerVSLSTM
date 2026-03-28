import numpy as np
import matplotlib.pyplot as plt

# 模拟训练epoch
epochs = 50
x = np.arange(1, epochs + 1)

# 模拟训练loss
train_loss = np.exp(-x / 15) + np.random.normal(0, 0.02, epochs)

# 模拟验证loss
val_loss = np.exp(-x / 14) + 0.05 + np.random.normal(0, 0.02, epochs)

# 绘图
plt.figure(figsize=(8,5))

plt.plot(x, train_loss, label='Train Loss', linewidth=2)
plt.plot(x, val_loss, label='Validation Loss', linestyle='--')

plt.title("Training and Validation Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

# 保存图片
plt.savefig("loss_curve.png", dpi=300)

plt.show()