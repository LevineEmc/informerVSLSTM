import numpy as np
import matplotlib.pyplot as plt

pred = np.load('results/informer_ETTh1_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp96_0/pred.npy')
true = np.load('results/informer_ETTh1_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp96_0/true.npy')

plt.figure(figsize=(10,5))
plt.plot(true[0,:,0], label='True')
plt.plot(pred[0,:,0], label='Pred')
plt.legend()
plt.title("Informer Forecast Result")
plt.show()