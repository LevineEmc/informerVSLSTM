import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data\\ETT\\ETTh1.csv")

plt.plot(df["OT"])
plt.title("Transformer Oil Temperature (ETT Dataset)")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.show()