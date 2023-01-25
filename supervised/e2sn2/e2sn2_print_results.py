import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, "..")
from results import e2_mp2_results, e2_lccsd_results, sn2_mp2_results, sn2_lccsd_results

result_functions = [e2_mp2_results, e2_lccsd_results, sn2_mp2_results, sn2_lccsd_results]
datasets = [("e2", "mp2"), ("e2", "lccsd"), ("sn2", "mp2"), ("sn2", "lccsd")]
minmax_tails = [(-10.0, 40.0), (-15.0, 45.0), (-10.0, 60.0), (-10.0, 60.0)]
colors = {-20.0: ["C0", "#113956"], 0.0: ["C1", "#b25503"], 20.0: ["C2", "#186618"], 40.0: ["C3", "#871616"], 60.0: ["C4", "#49266b"]}
#135b63
fraction_labels = ["min_max_tar", "min_max_tar_ran"]
fraction_titles = [None, "Exploration"]

iterations = [i for i in range(1, 40)]

for result_function, dataset, minmax_tail in zip(result_functions, datasets, minmax_tails):
	dataframe = pd.read_csv(dataset[0] + "_barriers_groups_" + dataset[1] + ".csv")
	min_barrier = min(dataframe["barrier"])
	max_barrier = max(dataframe["barrier"])

	min_results, max_results, target_results = result_function()

	# Plot found target barriers
	for fraction_label, fraction_title in zip(fraction_labels, fraction_titles):
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
		plt.title("ML " + dataset[0].title() + " (" + dataset[1] + "): Found target barriers" + ((" (" + fraction_title + ")") if fraction_title is not None else ""))
		plt.xlabel("Number of samples")
		plt.xticks(ticks=[12 * iter + 12 for iter in iterations if iter % 4 == 1])
		plt.ylabel("Activation barrier / kcal mol$^{-1}$")
		plt.legend(handles=handles[::-1], loc="upper right")
		plt.savefig(dataset[0] + "_" + dataset[1] + "_" + fraction_label + "_target.png", dpi=600)
		plt.clf()

	# Plot the minimum and maximum found barriers
	for fraction_label, fraction_title in zip(fraction_labels, fraction_titles):
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
			plt.plot([12, 490], [minmax_tail[0], minmax_tail[0]], "--", linewidth=1.0, color="grey")
			plt.plot([12, 490], [max_barrier, max_barrier], "--", linewidth=1.0, color="black")
			plt.plot([12, 490], [minmax_tail[1], minmax_tail[1]], "--", linewidth=1.0, color="grey")
		plt.title("ML " + dataset[0].title() + " (" + dataset[1] + "): Found min and max barriers" + ((" (" + fraction_title + ")") if fraction_title is not None else ""))
		plt.xlabel("Number of samples")
		plt.xticks(ticks=[12 * iter + 12 for iter in iterations if iter % 4 == 1])
		plt.ylabel("Activation barrier / kcal mol$^{-1}$")
		plt.legend(loc="center right")
		plt.savefig(dataset[0] + "_" + dataset[1] + "_" + fraction_label + "_minmax.png", dpi=600)
		plt.clf()
