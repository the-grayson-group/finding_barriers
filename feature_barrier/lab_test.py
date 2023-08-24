import numpy as np
from sklearn.linear_model import Ridge
from reaction_repr import generate_data, mkswp_vaska
from model_evaluation import fit_model

def find_target(model, Xy, parameter_ranges, target, starting_barrier):
	start_differences = np.abs(Xy[:,-1] - starting_barrier)
	start_index = start_differences.argsort()[0]
	train_set = Xy[start_index].copy().reshape(1,Xy.shape[1])
	test_set = np.delete(Xy.copy(), [start_index], axis=0)
	target_number = 1
	while train_set[~np.isnan(train_set[:,-1])].shape[0] < 5:
		n_sample = 5 - np.sum(~np.isnan(train_set[:,-1]))
		select_indices = np.random.randint(0, test_set.shape[0], size=(n_sample,))
		train_set = np.concatenate((train_set, test_set[select_indices,:]), axis=0)
		target_number += n_sample
		for i in select_indices:
			if np.isnan(test_set[i,-1]):
				for j in range(test_set.shape[1]):
					print("%.1f" % test_set[i,j], end=" ")
				print()
		test_set = np.delete(test_set, select_indices, axis=0)
		train_set = np.delete(train_set, np.where(np.isnan(train_set[:,-1]))[0], axis=0)
	print(target_number)
	for i in range(train_set.shape[0]):
		for j in range(train_set.shape[1]):
			print("%.1f" % train_set[i,j], end=" ")
		print()
	found = False
	while not found:
		trained_model = fit_model(model, train_set, parameter_ranges)
		predictions = trained_model.predict(test_set[:,:-1])
		prediction_differences = np.abs(predictions - target)
		c = 0
		to_delete = []
		sorted_indices = prediction_differences.argsort()
		closest_target_index = sorted_indices[c].astype(int)
		target_number += 1
		for i in range(test_set.shape[1]):
			print("%.1f" % test_set[closest_target_index][i], end=" ")
		print()
		while np.isnan(test_set[closest_target_index,-1]):
			to_delete.append(closest_target_index)
			c += 1
			closest_target_index = sorted_indices[c].astype(int)
			target_number += 1
			for i in range(test_set.shape[1]):
				print("%.1f" % test_set[closest_target_index][i], end=" ")
			print()
		train_set = np.concatenate((train_set, test_set[closest_target_index].reshape(1, test_set.shape[1])), axis=0)
		test_set = np.delete(test_set, [closest_target_index] + to_delete, axis=0)
		if np.min(np.abs(train_set[:,-1] - target)) < 1.0:
			found = True
			print(target_number)

if __name__ == "__main__":
	Xy = generate_data("vaska_barrier_feature_groups.csv", mkswp_vaska)
	np.random.seed(1)
	np.random.shuffle(Xy)
	model = Ridge()
	parameter_ranges = {"alpha": [10**i for i in range(-7,3)]}
	target = np.min(Xy[np.where(~np.isnan(Xy[:,-1]))[0],-1])
	starting_barrier = 20.0
	find_target(model, Xy, parameter_ranges, target, starting_barrier)
