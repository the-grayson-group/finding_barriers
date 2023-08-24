import multiprocessing
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge
from reaction_repr import generate_data, mkswp_ma, mkswp_vaska, mkswp_e2sn2
from model_evaluation import fit_model

def find_target(model, Xy, parameter_ranges, target):
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
		predictions, std_devs = trained_model.predict(test_set[:,:-1], return_std=True)
		# Negative prediction differences for maximizing expected improvement
		prediction_differences = -np.abs(predictions - target)
		best_barrier = np.max(-np.abs(train_set[:,-1] - target))
		z = (prediction_differences - best_barrier) / (std_devs + 1e-9)
		expected_improvements = (prediction_differences - best_barrier) * norm.cdf(z) + std_devs * norm.pdf(z)
		c = 0
		to_delete = []
		sorted_indices = expected_improvements.argsort()[::-1]
		most_promising_index = sorted_indices[c].astype(int)
		target_number += 1
		while np.isnan(test_set[most_promising_index,-1]):
			to_delete.append(most_promising_index)
			c += 1
			most_promising_index = sorted_indices[c].astype(int)
			target_number += 1
		train_set = np.concatenate((train_set, test_set[most_promising_index].reshape(1, test_set.shape[1])), axis=0)
		test_set = np.delete(test_set, [most_promising_index] + to_delete, axis=0)
		if np.min(np.abs(train_set[:,-1] - target)) < 1.0:
			found = True
	return target_number

def run_bayes_search(dataset, model, parameter_ranges, symmetry_func, n_repeats):
	dataset_file = open(dataset.replace("_groups.csv", "_bayes_results.txt"), "w")
	dataset_file.write(dataset + "\n")
	dataset_file.write("#" * 20 + "\n")
	dataset_file.flush()
	Xy = generate_data(dataset, symmetry_func)
	target_indices = set()
	barriers = Xy[np.where(~np.isnan(Xy[:,-1]))[0],-1]
	min_barrier = np.min(barriers)
	while min_barrier < np.max(barriers):
		differences = np.abs(min_barrier - barriers)
		target_indices.add(np.where(differences == np.min(differences))[0][0])
		min_barrier += 2.0
	target_indices.add(np.where(barriers == np.max(barriers))[0][0])
	target_indices = list(target_indices)
	targets = barriers[target_indices]
	targets = np.sort(targets)

	target_means = np.zeros(len(targets))
	target_stds = np.zeros(len(targets))

	for i, target in enumerate(targets):
		for seed in range(1, n_repeats+1):
			np.random.seed(seed)
			np.random.shuffle(Xy)
			target_number = find_target(model, Xy, parameter_ranges, target)
			dataset_file.write("target = %f seed = %d target_number = %d" % (target, seed, target_number) + "\n")
			dataset_file.flush()
			target_means[i] += target_number
			target_stds[i] += target_number**2

	target_means /= n_repeats
	target_stds = np.sqrt(np.abs((target_stds / n_repeats) - (target_means)**2))
	dataset_file.write("target: " + " ".join(str(tm) for tm in target_means) + "\n")
	dataset_file.write("targetstd: " + " ".join(str(tstd) for tstd in target_stds) + "\n")
	dataset_file.close()


if __name__ == "__main__":
	datasets = ["barrier_feature_groups.csv", "vaska_barrier_feature_groups.csv", "e2_barrier_feature_groups.csv", "sn2_barrier_feature_groups.csv"]
	symmetry_functions = [mkswp_ma, mkswp_vaska, mkswp_e2sn2, mkswp_e2sn2]
	model = BayesianRidge()
	parameter_ranges = {"alpha_init": [10**i for i in range(-7,3)], "lambda_init": [10**i for i in range(-7,3)]}
	n_repeats = 25
	for i in range(len(datasets)):
		p = multiprocessing.Process(target=run_bayes_search, args=(datasets[i], model, parameter_ranges, symmetry_functions[i], n_repeats))
		p.start()
