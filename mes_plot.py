import matplotlib.pyplot as plt

pred_lens = [24, 48, 96]

informer_mse = [0.586, 0.664, 0.921]
lstm_mse = [0.956, 1.047, 1.234]

plt.figure(figsize=(8,5))
plt.plot(pred_lens, informer_mse, marker='o', linewidth=2, label="Informer")
plt.plot(pred_lens, lstm_mse, marker='o', linewidth=2, label="LSTM")

plt.xlabel("Prediction Length")
plt.ylabel("MSE")
plt.title("MSE vs Prediction Length")
plt.legend()
plt.grid(True)
plt.show()