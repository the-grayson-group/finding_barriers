import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, "..")
from results import vaska_results

colors = {0.0: ["C0", "#113956"], 5.0: ["C1", "#b25503"], 10.0: ["C2", "#186618"], 15.0: ["C3", "#871616"], 20.0: ["C4", "#49266b"], 25.0: ["C9", "#135b63"]}
fraction_labels = ["min_max_tar", "min_max_tar_ran"]
fraction_titles = [None, "exploration"]

iterations = [i for i in range(1, 40)]

min_results, max_results, target_results = vaska_results()

for fraction_label, fraction_title in zip(fraction_labels, fraction_titles):
	dataframe = pd.read_csv("vaska_barriers_groups.csv")
	min_barrier = min(dataframe["barrier"])
	max_barrier = max(dataframe["barrier"])

	# Plot found target barriers
	handles = []
	for barrier in target_results[fraction_label].keys():
		found_target_values = []
		target_stds = []
		for iter in iterations:
			found_target_values.append(target_results[fraction_label][barrier][iter][0])
			target_stds.append(target_results[fraction_label][barrier][iter][1])
		handle, = plt.plot([12 * iter + 12 for iter in iterations], found_target_values, "o-", markersize=5.0, color=colors[barrier][0], label=str(round(barrier))+" kcal/mol")
		handles.append(handle)
		plt.fill_between([12 * iter + 12 for iter in iterations], found_target_values, [mean + std for mean, std in zip(found_target_values, target_stds)], color=colors[barrier][0], alpha=0.3)
		plt.fill_between([12 * iter + 12 for iter in iterations], found_target_values, [mean - std for mean, std in zip(found_target_values, target_stds)], color=colors[barrier][0], alpha=0.3)
		plt.plot([12, 490], [barrier, barrier], "--", linewidth=1.0, color=colors[barrier][1])
	plt.title("ML H$_2$: Found target barriers" + ((" (" + fraction_title + ")") if fraction_title is not None else ""))
	plt.xlabel("Number of samples")
	plt.xticks(ticks=[12 * iter + 12 for iter in iterations if iter % 4 == 1])
	plt.ylabel("Activation barrier / kcal mol$^{-1}$")
	plt.legend(handles=handles[::-1], loc="center right")
	plt.savefig("vaska_" + fraction_label + "_target.png", dpi=600)
	plt.clf()

	# Plot the minimum and maximum found barriers
	for barrier in min_results[fraction_label].keys():
		found_min_values = []
		min_stds = []
		found_max_values = []
		max_stds = []
		for iter in iterations:
			found_min_values.append(min_results[fraction_label][barrier][iter][0])
			min_stds.append(min_results[fraction_label][barrier][iter][1])
			found_max_values.append(max_results[fraction_label][barrier][iter][0])
			max_stds.append(max_results[fraction_label][barrier][iter][1])
		plt.plot([12 * iter + 12 for iter in iterations], found_min_values, "o-", markersize=5.0, color=colors[barrier][0], label=str(round(barrier))+" kcal/mol")
		plt.plot([12 * iter + 12 for iter in iterations], found_max_values, "X-", markersize=5.0, color=colors[barrier][0])
		plt.fill_between([12 * iter + 12 for iter in iterations], found_min_values, [mean + std for mean, std in zip(found_min_values, min_stds)], color=colors[barrier][0], alpha=0.3)
		plt.fill_between([12 * iter + 12 for iter in iterations], found_min_values, [mean - std for mean, std in zip(found_min_values, min_stds)], color=colors[barrier][0], alpha=0.3)
		plt.fill_between([12 * iter + 12 for iter in iterations], found_max_values, [mean + std for mean, std in zip(found_max_values, max_stds)], color=colors[barrier][0], alpha=0.3)
		plt.fill_between([12 * iter + 12 for iter in iterations], found_max_values, [mean - std for mean, std in zip(found_max_values, max_stds)], color=colors[barrier][0], alpha=0.3)
		plt.plot([12, 490], [min_barrier, min_barrier], "--", linewidth=1.0, color="black")
		plt.plot([12, 490], [2.0, 2.0], "--", linewidth=1.0, color="grey")
		plt.plot([12, 490], [max_barrier, max_barrier], "--", linewidth=1.0, color="black")
		plt.plot([12, 490], [24.0, 24.0], "--", linewidth=1.0, color="grey")
	plt.title("ML H$_2$: Found min and max barriers" + ((" (" + fraction_title + ")") if fraction_title is not None else ""))
	plt.xlabel("Number of samples")
	plt.xticks(ticks=[12 * iter + 12 for iter in iterations if iter % 4 == 1])
	plt.ylabel("Activation barrier / kcal mol$^{-1}$")
	plt.legend()
	plt.savefig("vaska_" + fraction_label + "_minmax.png", dpi=600)
	plt.clf()
