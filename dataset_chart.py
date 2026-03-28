import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Data/ETT/ETTh1.csv")

plt.figure(figsize=(10,4))
plt.plot(data['OT'][:1000])
plt.title("Time Series Example from ETTh1 Dataset")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()