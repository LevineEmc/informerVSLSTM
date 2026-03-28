import torch
import torch.nn as nn
import numpy as np
from data.data_loader import Dataset_ETT_hour
from torch.utils.data import DataLoader
from lstm_model import LSTMModel
from sklearn.metrics import mean_squared_error, mean_absolute_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数
seq_len = 96
pred_len = 96
batch_size = 32

# 数据集
train_set = Dataset_ETT_hour(
    root_path='./data/ETT/',
    data_path='ETTh1.csv',
    flag='train',
    size=[seq_len, 48, pred_len],
    features='M',
    target='OT',
    inverse=False
)

test_set = Dataset_ETT_hour(
    root_path='./data/ETT/',
    data_path='ETTh1.csv',
    flag='test',
    size=[seq_len, 48, pred_len],
    features='M',
    target='OT',
    inverse=False
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 模型
model = LSTMModel(
    input_dim=7,
    hidden_dim=128,
    num_layers=2,
    seq_len=seq_len,
    pred_len=pred_len,
    output_dim=7
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
for epoch in range(5):
    model.train()
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y[:, -pred_len:, :].float().to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 测试
model.eval()
preds = []
trues = []

with torch.no_grad():
    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y[:, -pred_len:, :].float().to(device)

        outputs = model(batch_x)

        preds.append(outputs.cpu().numpy())
        trues.append(batch_y.cpu().numpy())

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)

mse = mean_squared_error(trues.reshape(-1, 7), preds.reshape(-1, 7))
mae = mean_absolute_error(trues.reshape(-1, 7), preds.reshape(-1, 7))

print("LSTM Results:")
print("MSE:", mse)
print("MAE:", mae)
# 保存预测结果
np.save("lstm_pred_96.npy", preds)
np.save("lstm_true_96.npy", trues)