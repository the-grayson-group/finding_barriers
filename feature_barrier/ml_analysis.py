import multiprocessing
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from reaction_repr import generate_data, mkswp_ma, mkswp_vaska
from model_evaluation import fit_model

def report_model_performance(trained_model, train_set, test_set):
	test_set = test_set[np.where(~np.isnan(test_set[:,-1]))[0]]
	train_predictions = trained_model.predict(train_set[:,:-1])
	test_predictions = trained_model.predict(test_set[:,:-1])
	train_error = mean_absolute_error(train_set[:,-1], train_predictions)
	test_error = mean_absolute_error(test_set[:,-1], test_predictions)
	return train_error, test_error

def report_feature_importances(trained_model, train_set, test_set):
	test_set = test_set[np.where(~np.isnan(test_set[:,-1]))[0]]
	train_importances = permutation_importance(trained_model, train_set[:,:-1], train_set[:,-1], scoring="neg_mean_absolute_error", n_repeats=25)
	test_importances = permutation_importance(trained_model, test_set[:,:-1], test_set[:,-1], scoring="neg_mean_absolute_error", n_repeats=25)
	return train_importances.importances_mean, test_importances.importances_mean

def find_target(model, Xy, parameter_ranges, target, train_error_means, test_error_means, train_importance_means, test_importance_means, index_counts):
	train_set = Xy[:5].copy()
	test_set = Xy[5:].copy()
	target_number = 5
	while train_set[~np.isnan(train_set[:,-1])].shape[0] < 5:
		n_sample = 5 - np.sum(~np.isnan(train_set[:,-1]))
		select_indices = np.random.randint(0, test_set.shape[0], size=(n_sample,))
		train_set = np.concatenate((train_set, test_set[select_indices,:]), axis=0)
		target_number += n_sample
		test_set = np.delete(test_set, select_indices, axis=0)
		train_set = np.delete(train_set, np.where(np.isnan(train_set[:,-1]))[0], axis=0)
	found = False
	while not found:
		trained_model = fit_model(model, train_set, parameter_ranges)
		train_error, test_error = report_model_performance(trained_model, train_set, test_set)
		train_importances, test_importances = report_feature_importances(trained_model, train_set, test_set)
		if len(index_counts) <= train_set.shape[0] - 5:
			train_error_means.append(train_error)
			test_error_means.append(test_error)
			train_importance_means.append(train_importances)
			test_importance_means.append(test_importances)
			index_counts.append(1)
		else:
			train_error_means[train_set.shape[0] - 5] += train_error
			test_error_means[train_set.shape[0] - 5] += test_error
			train_importance_means[train_set.shape[0] - 5] += train_importances
			test_importance_means[train_set.shape[0] - 5] += test_importances
			index_counts[train_set.shape[0] - 5] += 1
		predictions = trained_model.predict(test_set[:,:-1])
		prediction_differences = np.abs(predictions - target)
		c = 0
		to_delete = []
		sorted_indices = prediction_differences.argsort()
		closest_target_index = sorted_indices[c].astype(int)
		target_number += 1
		while np.isnan(test_set[closest_target_index,-1]):
			to_delete.append(closest_target_index)
			c += 1
			closest_target_index = sorted_indices[c].astype(int)
			target_number += 1
		train_set = np.concatenate((train_set, test_set[closest_target_index].reshape(1, test_set.shape[1])), axis=0)
		test_set = np.delete(test_set, [closest_target_index] + to_delete, axis=0)
		if np.min(np.abs(train_set[:,-1] - target)) < 1.0:
			found = True
	return target_number

def run_ml_search(dataset, model, parameter_ranges, symmetry_func, n_repeats):
	dataset_file = open(dataset.replace("_groups.csv", "_analysis_results.txt"), "w")
	dataset_file.write(dataset + "\n")
	dataset_file.write("#" * 20 + "\n")
	Xy = generate_data(dataset, symmetry_func)
	np.random.seed(1)
	np.random.shuffle(Xy)
	target_indices = set()
	barriers = Xy[np.where(~np.isnan(Xy[:,-1]))[0],-1]
	min_barrier = np.min(barriers)
	max_barrier = np.max(barriers)
	for target_value in (min_barrier, 0.25 * (min_barrier + max_barrier), 0.5 * (min_barrier + max_barrier), 0.75 * (min_barrier + max_barrier), max_barrier):
		differences = np.abs(target_value - barriers)
		target_indices.add(np.where(differences == np.min(differences))[0][0])
	target_indices = list(target_indices)
	targets = barriers[target_indices]
	targets = np.sort(targets)

	for i, target in enumerate(targets):
		dataset_file.write("Target = %f\n" % target)
		train_error_means = []
		test_error_means = []
		train_importance_means = []
		test_importance_means = []
		index_counts = []
		for seed in range(1, n_repeats+1):
			np.random.seed(seed)
			np.random.shuffle(Xy)
			target_number = find_target(model, Xy, parameter_ranges, target, train_error_means, test_error_means, train_importance_means, test_importance_means, index_counts)
		for i in range(len(index_counts)):
			train_error_means[i] /= index_counts[i]
			test_error_means[i] /= index_counts[i]
			train_importance_means[i] /= index_counts[i]
			test_importance_means[i] /= index_counts[i]
		for i in range(len(index_counts)):
			dataset_file.write("train error = %f test error = %f\n" % (train_error_means[i], test_error_means[i]))
			dataset_file.write("train imports = %s\n" % " ".join([str(imp) for imp in train_importance_means[i]]))
			dataset_file.write("test imports = %s\n" % " ".join([str(imp) for imp in test_importance_means[i]]))
		dataset_file.flush()
	dataset_file.close()


if __name__ == "__main__":
	datasets = ["barrier_feature_groups.csv", "vaska_barrier_feature_groups.csv"]
	symmetry_functions = [mkswp_ma, mkswp_vaska]
	model = Ridge()
	parameter_ranges = {"alpha": [10**i for i in range(-7,3)]}
	n_repeats = 25
	for i in range(len(datasets)):
		p = multiprocessing.Process(target=run_ml_search, args=(datasets[i], model, parameter_ranges, symmetry_functions[i], n_repeats))
		p.start()
