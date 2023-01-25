import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("barriers_groups.csv")
data.drop("Unnamed: 0", axis=1, inplace=True)
plt.hist(data["barrier"]*627.503, bins=100)
plt.ylabel("Frequency", fontsize=12)
plt.xlabel("Activation barrier / kcal mol$^{-1}$", fontsize=12)
plt.title("Michael addition barrier distribution", fontsize=12)
plt.savefig("barrier_dist.png", dpi=600)
