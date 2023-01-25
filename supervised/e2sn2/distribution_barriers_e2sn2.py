import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("e2_barriers_groups_mp2.csv")
plt.hist(data["barrier"], bins=100, color="C1")
plt.ylabel("Frequency", fontsize=12)
plt.xlabel("Activation barrier / kcal mol$^{-1}$", fontsize=12)
plt.title("E2 (con, mp2) barrier distribution", fontsize=12)
plt.savefig("e2_barrier_dist_mp2.png", dpi=600)
plt.clf()

data = pd.read_csv("sn2_barriers_groups_mp2.csv")
plt.hist(data["barrier"], bins=100, color="C3")
plt.ylabel("Frequency", fontsize=12)
plt.xlabel("Activation barrier / kcal mol$^{-1}$", fontsize=12)
plt.title("S$_N$2 (con, mp2) barrier distribution", fontsize=12)
plt.savefig("sn2_barrier_dist_mp2.png", dpi=600)
plt.clf()

data = pd.read_csv("e2_barriers_groups_lccsd.csv")
plt.hist(data["barrier"], bins=100, color="C1")
plt.ylabel("Frequency", fontsize=12)
plt.xlabel("Activation barrier / kcal mol$^{-1}$", fontsize=12)
plt.title("E2 (uncon, LCCSD/KRR) barrier distribution", fontsize=12)
plt.savefig("e2_barrier_dist_lccsd.png", dpi=600)
plt.clf()

data = pd.read_csv("sn2_barriers_groups_lccsd.csv")
plt.hist(data["barrier"], bins=100, color="C3")
plt.ylabel("Frequency", fontsize=12)
plt.xlabel("Activation barrier / kcal mol$^{-1}$", fontsize=12)
plt.title("S$_N$2 (uncon, LCCSD/KRR) barrier distribution", fontsize=12)
plt.savefig("sn2_barrier_dist_lccsd.png", dpi=600)
plt.clf()
