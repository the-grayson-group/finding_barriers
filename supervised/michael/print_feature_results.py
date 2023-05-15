import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, "..")
from feature_results import ma_feature_results

colors = {10.0: ["C0", "#113956"], 20.0: ["C1", "#b25503"], 30.0: ["C2", "#186618"], 40.0: ["C3", "#871616"], 50.0: ["C4", "#49266b"]}
fraction_labels = ["feature_min_max_tar"]
fraction_titles = [None]

iterations = [i for i in range(1, 40)]

min_results, max_results, target_results = ma_feature_results()

for fraction_label, fraction_title in zip(fraction_labels, fraction_titles):
	dataframe = pd.read_csv("barriers_groups.csv")
	min_barrier = min(dataframe["barrier"]) * 627.503
	max_barrier = max(dataframe["barrier"]) * 627.503

	# Plot found target barriers
	handles = []
	fig = plt.figure()
	ax = fig.add_subplot()
	for barrier in target_results[fraction_label].keys():
		found_target_values = []
		target_stds = []
		for iter in iterations:
			found_target_values.append(target_results[fraction_label][barrier][iter][0])
			target_stds.append(target_results[fraction_label][barrier][iter][1])
		handle, = ax.plot([12 * iter + 12 for iter in iterations], found_target_values, "o-", markersize=5.0, color=colors[barrier][0], label=str(round(barrier))+" kcal/mol")
		handles.append(handle)
		ax.fill_between([12 * iter + 12 for iter in iterations], found_target_values, [mean + std for mean, std in zip(found_target_values, target_stds)], color=colors[barrier][0], alpha=0.3)
		ax.fill_between([12 * iter + 12 for iter in iterations], found_target_values, [mean - std for mean, std in zip(found_target_values, target_stds)], color=colors[barrier][0], alpha=0.3)
		ax.plot([12, 490], [barrier, barrier], "--", linewidth=1.0, color=colors[barrier][1])
	ax.set_title("ML MA (features): Found target barriers" + ((" (" + fraction_title + ")") if fraction_title is not None else ""))
	ax.set_xlabel("Number of samples")
	ax.set_xticks(ticks=[12 * iter + 12 for iter in iterations if iter % 4 == 1])
	ax.set_ylabel("Activation barrier / kcal mol$^{-1}$")
	lgd = ax.legend(handles=handles[::-1], bbox_to_anchor=(1.04, 1))
	fig.savefig("ma_" + fraction_label + "_target.png", bbox_extra_artists=(lgd,), bbox_inches="tight", dpi=600)
	plt.gcf().clear()

	# Plot the minimum and maximum found barriers
	fig = plt.figure()
	ax = fig.add_subplot()
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
		ax.plot([12 * iter + 12 for iter in iterations], found_min_values, "o-", markersize=5.0, color=colors[barrier][0], label=str(round(barrier))+" kcal/mol")
		ax.plot([12 * iter + 12 for iter in iterations], found_max_values, "X-", markersize=5.0, color=colors[barrier][0])
		ax.fill_between([12 * iter + 12 for iter in iterations], found_min_values, [mean + std for mean, std in zip(found_min_values, min_stds)], color=colors[barrier][0], alpha=0.3)
		ax.fill_between([12 * iter + 12 for iter in iterations], found_min_values, [mean - std for mean, std in zip(found_min_values, min_stds)], color=colors[barrier][0], alpha=0.3)
		ax.fill_between([12 * iter + 12 for iter in iterations], found_max_values, [mean + std for mean, std in zip(found_max_values, max_stds)], color=colors[barrier][0], alpha=0.3)
		ax.fill_between([12 * iter + 12 for iter in iterations], found_max_values, [mean - std for mean, std in zip(found_max_values, max_stds)], color=colors[barrier][0], alpha=0.3)
		ax.plot([12, 490], [min_barrier, min_barrier], "--", linewidth=1.0, color="black")
		ax.plot([12, 490], [15.0, 15.0], "--", linewidth=1.0, color="grey")
		ax.plot([12, 490], [max_barrier, max_barrier], "--", linewidth=1.0, color="black")
		ax.plot([12, 490], [45.0, 45.0], "--", linewidth=1.0, color="grey")
	ax.set_title("ML MA (features): Found min and max barriers" + ((" (" + fraction_title + ")") if fraction_title is not None else ""))
	ax.set_xlabel("Number of samples")
	ax.set_xticks(ticks=[12 * iter + 12 for iter in iterations if iter % 4 == 1])
	ax.set_ylabel("Activation barrier / kcal mol$^{-1}$")
	lgd = ax.legend(bbox_to_anchor=(1.04, 1))
	fig.savefig("ma_" + fraction_label + "_minmax.png", bbox_extra_artists=(lgd,), bbox_inches="tight", dpi=600)
	plt.gcf().clear()
