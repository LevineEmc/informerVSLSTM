import numpy as np
import matplotlib.pyplot as plt

# 读取 Informer 结果
inf_pred = np.load("results/informer_ETTh1_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp96_0/pred.npy")
inf_true = np.load("results/informer_ETTh1_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp96_0/true.npy")

# 读取 LSTM 结果
lstm_pred = np.load("lstm_pred_96.npy")
lstm_true = np.load("lstm_true_96.npy")

# 取第一条样本，第一个变量
plt.figure(figsize=(12,5))

plt.plot(inf_true[0,:,0], label="True", linewidth=2)
plt.plot(inf_pred[0,:,0], label="Informer")
plt.plot(lstm_pred[0,:,0], label="LSTM")

plt.title("Prediction Comparison (pred_len=96)")
plt.legend()
plt.show()