import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("vaska_barriers_groups.csv")
plt.hist(data["barrier"], bins=100, color="C2")
plt.ylabel("Frequency", fontsize=12)
plt.xlabel("Activation barrier / kcal mol$^{-1}$", fontsize=12)
plt.title("H$_2$ activation barrier distribution", fontsize=12)
plt.savefig("vaska_barrier_dist.png", dpi=600)
plt.clf()
