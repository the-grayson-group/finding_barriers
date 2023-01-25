import csv
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

def barriers_groups_lccsd(data_file, csv_file):
	data_file = open(data_file, "r")
	csv_file = open(csv_file, "w", newline="")
	csv_writer = csv.writer(csv_file, delimiter=",")
	csv_writer.writerow(["R1", "R2", "R3", "R4", "X", "Y", "barrier"])
	for line in data_file:
		mol_data = line.split(" ")[0].split("_")
		mol_data.append(float(line.split(" ")[-1].rstrip("\n")))
		csv_writer.writerow(mol_data)
	data_file.close()
	csv_file.close()

def barriers_groups_mp2(data_file, reaction, csv_file):
	data_file = open(data_file, "r")
	csv_file = open(csv_file, "w", newline="")
	csv_writer = csv.writer(csv_file, delimiter=",")
	csv_writer.writerow(["R1", "R2", "R3", "R4", "X", "Y", "barrier"])
	for line in data_file:
		if reaction in line and "mp2" in line and "-constrained-" in line:
			mol_data = line.split(",")[0].split("_")
			mol_data.append(float(line.split(",")[-2]))
			csv_writer.writerow(mol_data)
	data_file.close()
	csv_file.close()

def get_data(csv_file, target_value=0.0, rstate=1):
	data = pd.read_csv(csv_file)
	# Create the one-hot encoded data
	R1_len = len(set(data["R1"]))
	R2_len = len(set(data["R2"]))
	R3_len = len(set(data["R3"]))
	R4_len = len(set(data["R4"]))
	X_len = len(set(data["X"]))
	Y_len = len(set(data["Y"]))
	R1_inc = 0
	R2_inc = R1_len
	R3_inc = R2_len + R2_inc
	R4_inc = R3_len + R3_inc
	X_inc = R4_len + R4_inc
	Y_inc = X_len + X_inc
	R1_indices = {}
	R2_indices = {}
	R3_indices = {}
	R4_indices = {}
	X_indices = {}
	Y_indices = {}
	Xy = np.zeros((len(data), Y_inc + Y_len + 1))
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
		if data["X"][i] not in X_indices:
			X_indices[data["X"][i]] = X_inc
			X_inc += 1
		Xy[i,X_indices[data["X"][i]]] = 1
		if data["Y"][i] not in Y_indices:
			Y_indices[data["Y"][i]] = Y_inc
			Y_inc += 1
		Xy[i,Y_indices[data["Y"][i]]] = 1
		# Add the barrier in kcal/mol to the data matrix
		Xy[i,-1] = data["barrier"][i]
	# Shuffle the data
	np.random.seed(rstate)
	np.random.shuffle(Xy)
	# Move the minimum, maximum closest-to-target barriers to the end of the dataset so they won't be sampled immediately
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


#barriers_groups_lccsd("e2_lccsd_data.txt", "e2_barriers_groups_lccsd.csv")
#barriers_groups_mp2("barriers.txt", "e2", "e2_barriers_groups_mp2.csv")
#barriers_groups_lccsd("sn2_lccsd_data.txt", "sn2_barriers_groups_lccsd.csv")
#barriers_groups_mp2("barriers.txt", "sn2", "sn2_barriers_groups_mp2.csv")

datasets = [("e2", "mp2"), ("e2", "lccsd"), ("sn2", "mp2"), ("sn2", "lccsd")]

iterations = [i for i in range(1, 40)]
barriers = [-20.0, 0.0, 20.0, 40.0, 60.0]
fractions = [(1/3, 1/3, 0.0), (0.25, 0.25, 0.25)]
fraction_labels = ["min_max_tar", "min_max_tar_ran"]

for dataset in datasets:
	min_results = {}
	max_results = {}
	target_results = {}
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
				print(dataset[0], dataset[1], label, target, "kcal/mol", "rstate =", rstate)
				Xy = get_data(dataset[0] + "_barriers_groups_" + dataset[1] + ".csv", target_value=target, rstate=rstate)
				min_values, max_values, target_values = find_min_max_target(Xy, iterations, sample_size=12, minmax_fraction=fraction[0], target_fraction=fraction[1], random_fraction=fraction[2], target=target)
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

	results_file = open("../results.py", "a")
	results_file.write("def " + dataset[0] + "_" + dataset[1] + "_results():\n")
	results_file.write("\tmin_results = " + str(min_results) + "\n")
	results_file.write("\tmax_results = " + str(max_results) + "\n")
	results_file.write("\ttarget_results = " + str(target_results) + "\n")
	results_file.write("\treturn min_results, max_results, target_results\n\n")
	results_file.close()
