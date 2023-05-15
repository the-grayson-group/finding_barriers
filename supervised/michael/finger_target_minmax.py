import os
import re
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from rdkit import Chem
from rdkit.Chem import AllChem

def get_fingerprints(csv_file, structure_dir, radius=3, n_bits=2048):
	file_regex = re.compile("GS-(\d{1,4})-\d\.mol")
	file_list = os.listdir(structure_dir)
	file_list.sort(key=lambda file: int(file_regex.search(file).group(1)))
	mols = []
	for file in file_list:
		mol = Chem.MolFromMol2File(structure_dir + "/" + file)
		mols.append(mol)
	Xy = np.zeros((len(mols), n_bits + 1))
	for m in range(len(mols)):
		fingerprint = AllChem.GetMorganFingerprintAsBitVect(mols[m], radius, nBits=n_bits)
		fp_array = np.zeros((0,))
		Chem.DataStructs.ConvertToNumpyArray(fingerprint, fp_array)
		Xy[m,:n_bits] = fp_array
	reaction_info = pd.read_csv(csv_file)
	Xy[:,-1] = reaction_info["barrier"] * 627.503
	return Xy

def get_data(Xy, target_value=0.0, rstate=1):
	# Shuffle the data
	np.random.seed(rstate)
	np.random.shuffle(Xy)
	# Move the minimum, maximum and closest-to-target barriers to the end of the dataset so they won't be sampled immediately
	min_barr_index = np.where(Xy[:,-1] == np.min(Xy[:,-1]))[0][0]
	max_barr_index = np.where(Xy[:,-1] == np.max(Xy[:,-1]))[0][0]
	target_barr_index = np.where(np.abs(Xy[:,-1] - target_value) == np.min(np.abs(Xy[:,-1] - target_value)))[0][0]
	min_sample = Xy[min_barr_index].copy()
	max_sample = Xy[max_barr_index].copy()
	target_sample = Xy[target_barr_index].copy()
	Xy[min_barr_index] = Xy[-1]
	Xy[max_barr_index] = Xy[-2]
	Xy[target_barr_index] = Xy[-3]
	Xy[-1] = min_sample
	Xy[-2] = max_sample
	Xy[-3] = target_sample
	return Xy

def fit_model(train_set):
	parameter_ranges = {"alpha": [10**i for i in range(-7, 3)], "gamma": [10**i for i in range(-7, 3)]}
	grid_search = GridSearchCV(KernelRidge(kernel="rbf"), parameter_ranges, scoring="neg_mean_absolute_error", cv=5, refit=True, n_jobs=-1)
	grid_search.fit(train_set[:,:-1], train_set[:,-1])
	return grid_search.best_estimator_

def find_min_max_target(Xy, iterations, sample_size=12, minmax_fraction=(1/3), target_fraction=(1/3), random_fraction=0.0, target=0.0):
	# Initialise training and testing sets
	train_set = Xy[:sample_size,:].copy()
	test_set = Xy[sample_size:,:].copy()
	print("Training set size: %d" % train_set.shape[0])
	print("Testing set size: %d" % test_set.shape[0])
	min_values = {}
	max_values = {}
	target_values = {}
	# Begin loop to find min, max and target reactions
	for i in range(1, iterations[-1] + 1):
		# Train the model on the training set and predict the barriers for the "testing" set
		model = fit_model(train_set)
		predictions = model.predict(test_set[:,:-1])
		# Find the smallest and largest predicted barriers
		if minmax_fraction > 0.0:
			sorted_prediction_indices = predictions.argsort()
			smallest_barrier_indices = sorted_prediction_indices[:round(sample_size * minmax_fraction)].astype(int)
			train_set = np.concatenate((train_set, test_set[smallest_barrier_indices,:]), axis=0)
			test_set = np.delete(test_set, smallest_barrier_indices, axis=0)
			predictions = np.delete(predictions, smallest_barrier_indices)
			sorted_prediction_indices = predictions.argsort()
			largest_barrier_indices = sorted_prediction_indices[-round(sample_size * minmax_fraction):].astype(int)
			train_set = np.concatenate((train_set, test_set[largest_barrier_indices,:]), axis=0)
			test_set = np.delete(test_set, largest_barrier_indices, axis=0)
			predictions = np.delete(predictions, largest_barrier_indices)
		# Find the barriers predicted closest to the target value
		prediction_differences = np.abs(predictions - target)
		closest_target_indices = prediction_differences.argsort()[:round(sample_size * target_fraction)].astype(int)
		train_set = np.concatenate((train_set, test_set[closest_target_indices,:]), axis=0)
		test_set = np.delete(test_set, closest_target_indices, axis=0)
		# Obtain an additional random sample
		if random_fraction > 0.0:
			test_indices = np.arange(test_set.shape[0])
			random_indices = np.random.choice(test_indices, size=round(sample_size * random_fraction), replace=False).astype(int)
			train_set = np.concatenate((train_set, test_set[random_indices,:]), axis=0)
			test_set = np.delete(test_set, random_indices, axis=0)
		if i in iterations:
			print("Iteration %d" % i)
			print("Minsize, maxsize, tarsize, ransize: %d %d %d %d" % (minmax_fraction * sample_size, minmax_fraction * sample_size, target_fraction * sample_size, random_fraction * sample_size))
			print("Training set size: %d" % train_set.shape[0])
			print("Testing set size: %d" % test_set.shape[0])
			# Analyse the train set
			found_target_value = train_set[:,-1][np.where(np.abs(train_set[:,-1] - target) == np.min(np.abs(train_set[:,-1] - target)))[0][0]]
			min_found_value = np.min(train_set[:,-1])
			max_found_value = np.max(train_set[:,-1])
			min_values[i] = min_found_value
			max_values[i] = max_found_value
			target_values[i] = found_target_value
			print("Min, max, target values: %.4f %.4f %.4f" % (min_found_value, max_found_value, found_target_value))
	return min_values, max_values, target_values


min_results = {}
max_results = {}
target_results = {}
barriers = [10.0, 20.0, 30.0, 40.0, 50.0]
iterations = [i for i in range(1, 40)]
fractions = [(1/3, 1/3, 0.0)]
fraction_labels = ["finger_min_max_tar"]

Xy = get_fingerprints("barriers_groups.csv", "reactant_structures", radius=3, n_bits=2048)

for label, fraction in zip(fraction_labels, fractions):
	min_results[label] = {}
	max_results[label] = {}
	target_results[label] = {}
	for target in barriers:
		min_results[label][target] = {t: [0,0] for t in iterations}
		max_results[label][target] = {t: [0,0] for t in iterations}
		target_results[label][target] = {t: [0,0] for t in iterations}
		for rstate in range(1,51):
			print(80 * "-")
			print("MA", label, target, "kcal/mol", "rstate =", rstate)
			Xy_data = get_data(Xy.copy(), target_value=target, rstate=rstate)
			min_values, max_values, target_values = find_min_max_target(Xy_data, iterations, sample_size=12, minmax_fraction=fraction[0], target_fraction=fraction[1], random_fraction=fraction[2], target=target)
			for iteration in iterations:
				min_results[label][target][iteration][0] += min_values[iteration]
				min_results[label][target][iteration][1] += min_values[iteration]**2
				max_results[label][target][iteration][0] += max_values[iteration]
				max_results[label][target][iteration][1] += max_values[iteration]**2
				target_results[label][target][iteration][0] += target_values[iteration]
				target_results[label][target][iteration][1] += target_values[iteration]**2

		for iteration in iterations:
			min_results[label][target][iteration][0] /= 50
			min_results[label][target][iteration][1] = np.sqrt(np.abs((min_results[label][target][iteration][1] / 50) - min_results[label][target][iteration][0]**2))
			max_results[label][target][iteration][0] /= 50
			max_results[label][target][iteration][1] = np.sqrt(np.abs((max_results[label][target][iteration][1] / 50) - max_results[label][target][iteration][0]**2))
			target_results[label][target][iteration][0] /= 50
			target_results[label][target][iteration][1] = np.sqrt(np.abs((target_results[label][target][iteration][1] / 50) - target_results[label][target][iteration][0]**2))

results_file = open("../feature_results.py", "a")
results_file.write("def ma_finger_results():\n")
results_file.write("\tmin_results = " + str(min_results) + "\n")
results_file.write("\tmax_results = " + str(max_results) + "\n")
results_file.write("\ttarget_results = " + str(target_results) + "\n")
results_file.write("\treturn min_results, max_results, target_results\n\n")
results_file.close()
