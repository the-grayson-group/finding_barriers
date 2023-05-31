import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition as rdRGD
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

def get_reactions_and_barriers(csv_file):
	"""
	This function returns the one-hot encodings of each of the reactions in
	the defined reaction space, as well as the barriers of each of those
	reactions. In practise, one would not have access to all barriers from
	the very start, these would be measured one sample at a time, just before
	retraining the ML model.
	"""
	data = pd.read_csv(csv_file)
	# Create the one-hot encoded data
	R1_len = len(set(data["R1"]))
	R2_len = len(set(data["R2"]))
	R3_len = len(set(data["R3"]))
	R4_len = len(set(data["R4"]))
	R1_inc = 0
	R2_inc = R1_len
	R3_inc = R2_len + R2_inc
	R4_inc = R3_len + R3_inc
	R1_indices = {}
	R2_indices = {}
	R3_indices = {}
	R4_indices = {}
	Xy = np.zeros((len(data), R4_inc + R4_len + 1))
	for i in range(len(data)):
		if data["R1"][i] not in R1_indices:
			R1_indices[data["R1"][i]] = R1_inc
			R1_inc += 1
		Xy[i,R1_indices[data["R1"][i]]] = 1
		if data["R2"][i] not in R2_indices:
			R2_indices[data["R2"][i]] = R2_inc
			R2_inc += 1
		Xy[i,R2_indices[data["R2"][i]]] = 1
		if data["R3"][i] not in R3_indices:
			R3_indices[data["R3"][i]] = R3_inc
			R3_inc += 1
		Xy[i,R3_indices[data["R3"][i]]] = 1
		if data["R4"][i] not in R4_indices:
			R4_indices[data["R4"][i]] = R4_inc
			R4_inc += 1
		Xy[i,R4_indices[data["R4"][i]]] = 1
		Xy[i,-1] = data["barrier"][i]
	return Xy

def fit_model(train_set):
	parameter_ranges = {"alpha": [10**i for i in range(-7, 3)], "gamma": [10**i for i in range(-7, 3)]}
	grid_search = GridSearchCV(KernelRidge(kernel="rbf"), parameter_ranges, scoring="neg_mean_absolute_error", cv=5, refit=True, n_jobs=-1)
	grid_search.fit(train_set[:,:-1], train_set[:,-1])
	return grid_search.best_estimator_

def find_min_max_target(Xy, iterations, sample_size=6, target=0.0, starting_barrier=20.0, minmax_fraction=(1/3), target_fraction=(1/3)):
	# Initialising training set with the single barrier we found from experiment
	start_index = np.where(np.abs(Xy[:,-1] - starting_barrier) == np.min(np.abs(Xy[:,-1] - starting_barrier)))[0][0]
	print("Starting reaction: %d" % start_index)
	train_set = Xy[start_index].copy().reshape(1, Xy.shape[1])
	test_indices = [i for i in range(len(Xy))]
	test_indices.remove(start_index)
	# Add an extra sample of random barriers to get started
	sample_indices = np.random.choice(test_indices, size=sample_size-1, replace=False)
	train_set = np.concatenate((train_set, Xy[sample_indices]), axis=0)
	# Make the "test" set of all the other reactions
	for si in sample_indices:
		test_indices.remove(si)
	test_set = Xy[test_indices].copy()
	# Create dictionaries to store results
	min_values = {}
	max_values = {}
	target_values = {}
	print("Min, max, target values: %.4f %.4f %.4f" % (np.min(train_set[:,-1]), np.max(train_set[:,-1]), train_set[:,-1][np.where(np.abs(train_set[:,-1] - target) == np.min(np.abs(train_set[:,-1] - target)))[0][0]]))
	# Begin searching process
	for i in range(1, iterations+1):
		model = fit_model(train_set)
		predictions = model.predict(test_set[:,:-1])
		# Find the min and max predicted barriers
		if minmax_fraction > 0.0:
			sorted_prediction_indices = predictions.argsort()
			smallest_barrier_indices = sorted_prediction_indices[:round(sample_size * minmax_fraction)].astype(int)
			train_set = np.concatenate((train_set, test_set[smallest_barrier_indices]), axis=0)
			test_set = np.delete(test_set, smallest_barrier_indices, axis=0)
			predictions = np.delete(predictions, smallest_barrier_indices)
			sorted_prediction_indices = predictions.argsort()
			largest_barrier_indices = sorted_prediction_indices[-round(sample_size * minmax_fraction):].astype(int)
			train_set = np.concatenate((train_set, test_set[largest_barrier_indices]), axis=0)
			test_set = np.delete(test_set, largest_barrier_indices, axis=0)
			predictions = np.delete(predictions, largest_barrier_indices)
		# Find the barriers predicted closest to the target value
		prediction_differences = np.abs(predictions - target)
		closest_target_indices = prediction_differences.argsort()[:round(sample_size * target_fraction)].astype(int)
		train_set = np.concatenate((train_set, test_set[closest_target_indices]), axis=0)
		test_set = np.delete(test_set, closest_target_indices, axis=0)
		# Record results
		min_values[i] = np.min(train_set[:,-1])
		max_values[i] = np.max(train_set[:,-1])
		target_values[i] = train_set[:,-1][np.where(np.abs(train_set[:,-1] - target) == np.min(np.abs(train_set[:,-1] - target)))[0][0]]
		print("Iteration %d" % i)
		print("Minsize, maxsize, tarsize: %d %d %d" % (minmax_fraction * sample_size, minmax_fraction * sample_size, target_fraction * sample_size))
		print("Training set size: %d" % len(train_set))
		print("Testing set size: %d" % len(test_set))
		print("Min, max, target values: %.4f %.4f %.4f" % (min_values[i], max_values[i], target_values[i]))
		print("Range: %.4f" % (max_values[i] - min_values[i]))
		target_train_index = np.where(np.abs(train_set[:,-1] - target) == np.min(np.abs(train_set[:,-1] - target)))[0][0]
		target_index = np.where(np.all(Xy == train_set[target_train_index], axis=1))[0][0]
		print("Reaction index: %d" % (target_index + 2))
	return min_values, max_values, target_values

def plot_results(min_values, max_values, target_values, iterations, target, title, file, sample_size=6):
	# Plot the barriers that are closest to the desired value
	# as well as the estimated range of the barrier distribution
	fig, ax1 = plt.subplots()
	ax1.set_title(title)
	ax1.set_xlabel("Number of samples")
	ax1.set_ylabel("Difference from target barrier", color="C0")
	ax1.tick_params(axis="y", labelcolor="C0")
	ax2 = ax1.twinx()
	ax2.set_ylabel("Estimated barrier range", color="C3")
	ax2.tick_params(axis="y", labelcolor="C3")
	sample_sizes, min_barriers, max_barriers, target_barriers = [], [], [], []
	for i in range(1, iterations+1):
		sample_sizes.append(sample_size + sample_size * i)
		min_barriers.append(min_values[i])
		max_barriers.append(max_values[i])
		target_barriers.append(abs(target_values[i] - target))
	ax2.fill_between(sample_sizes, min_barriers, max_barriers, color="C3", alpha=0.3)
	ax2.plot([sample_size, sample_size * iterations + 8], [np.min(Xy[:,-1]), np.min(Xy[:,-1])], "--", linewidth=1.0, color="C3", alpha=0.5)
	ax2.plot([sample_size, sample_size * iterations + 8], [np.max(Xy[:,-1]), np.max(Xy[:,-1])], "--", linewidth=1.0, color="C3", alpha=0.5)
	ax1.plot([sample_size, sample_size * iterations + 8], [0.0, 0.0], "--", linewidth=1.0, color="C0", alpha=0.5)
	ax1.plot(sample_sizes, target_barriers, "x-", markersize=5.0, color="C0")
	ax1.set_xticks(sample_sizes)
	fig.savefig(file, bbox_inches="tight", dpi=600)
	plt.gcf().clear()


np.random.seed(5)
Xy = get_reactions_and_barriers("vaska_barriers_groups.csv")
min_values, max_values, target_values = find_min_max_target(Xy, 14, target=5.0)
plot_results(min_values, max_values, target_values, 14, 5.0, "Catalyst Optimization: Target = 5 kcal/mol", "real_test_min_max_tar5.png")
