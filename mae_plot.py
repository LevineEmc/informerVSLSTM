import matplotlib.pyplot as plt

pred_lens = [24, 48, 96]
informer_mae = [0.572, 0.606, 0.753]
lstm_mae = [0.732, 0.734, 0.846]

plt.figure(figsize=(8,5))
plt.plot(pred_lens, informer_mae, marker='o', linewidth=2, label='Informer')
plt.plot(pred_lens, lstm_mae, marker='o', linewidth=2, label='LSTM')
plt.xlabel('Prediction Length')
plt.ylabel('MAE')
plt.title('MAE vs Prediction Length')
plt.legend()
plt.grid(True)
plt.show()