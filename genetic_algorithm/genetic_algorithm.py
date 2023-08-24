# This implementation is heavily based on that of the following:
# J. Brownlee, Simple Genetic Algorithm From Scratch in Python, Machine Learning Mastery, Available from https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/, accessed July 2023.
# Along with the truncation procedure for integer variables from:
# K. Deep, K. P. Singh, M. L. Kansal, C. Mohan, A real coded genetic algorithm for solving integer and mixed integer optimization problems, Applied Mathematics and Computation, 212, 2009, 505-518, DOI: 10.1016/j.amc.2009.02.044.
import multiprocessing
import datetime
import numpy as np
from reaction_repr import generate_data, mkswp_ma, mkswp_vaska, mkswp_e2sn2

def difference_from_target(one_hot_encoding, Xy, target, swp_func, group_bitlens, invalid_penalty):
	index = ((Xy[:,:-1] == one_hot_encoding).all(axis=1) | (Xy[:,:-1] == swp_func(one_hot_encoding, group_bitlens)).all(axis=1)).nonzero()[0]
	barrier = Xy[index[0],-1]
	if np.isnan(barrier):
		return invalid_penalty
	else:
		return np.abs(target - barrier)

def swp_ma_onehot(one_hot_encoding, group_bitlens):
	return one_hot_encoding

def swp_e2sn2_onehot(one_hot_encoding, group_bitlens):
	one_hot_encoding = one_hot_encoding.copy()
	group_positions = [0] + list(np.cumsum(group_bitlens[:-1]))
	r1_string = one_hot_encoding[:group_bitlens[1]].copy()
	one_hot_encoding[:group_positions[1]] = one_hot_encoding[group_positions[1]:group_positions[2]]
	one_hot_encoding[group_positions[1]:group_positions[2]] = r1_string
	r3_string = one_hot_encoding[group_positions[2]:group_positions[3]].copy()
	one_hot_encoding[group_positions[2]:group_positions[3]] = one_hot_encoding[group_positions[3]:group_positions[4]]
	one_hot_encoding[group_positions[3]:group_positions[4]] = r3_string
	return one_hot_encoding

def swp_vaska_onehot(one_hot_encoding, group_bitlens):
	one_hot_encoding = one_hot_encoding.copy()
	group_positions = [0] + list(np.cumsum(group_bitlens[:-1]))
	r3_string = one_hot_encoding[group_positions[2]:group_positions[3]].copy()
	one_hot_encoding[group_positions[2]:group_positions[3]] = one_hot_encoding[group_positions[3]:]
	one_hot_encoding[group_positions[3]:] = r3_string
	return one_hot_encoding

def onehot_to_int(one_hot_encoding):
	return int("".join([str(int(c)) for c in one_hot_encoding]), 2)

def decode(bitstring, group_bitlens, n_bits, bounds):
	one_hot = np.zeros(sum(group_bitlens))
	point = 0
	for i in range(len(bounds)):
		largest = 2**n_bits
		start, end = i * n_bits, (i * n_bits) + n_bits
		binary_list = bitstring[start:end]
		group_code = onehot_to_int(binary_list)
		group_code = bounds[i][0] + (group_code / largest) * (bounds[i][1] - bounds[i][0])
		group_code = int(group_code) if np.random.rand() < 0.5 else int(group_code) + 1
		one_hot[point + group_code] = 1
		point += group_bitlens[i]
	return one_hot

def selection(pop, scores, k=3):
	selection_index = np.random.randint(len(pop))
	for i in np.random.randint(0, len(pop), k-1):
		if scores[i] < scores[selection_index]:
			selection_index = i
	return pop[selection_index]

def crossover(parent1, parent2, cross_rate):
	child1, child2 = parent1.copy(), parent2.copy()
	if np.random.rand() < cross_rate:
		point = np.random.randint(1, len(parent1)-2)
		child1[:point], child1[point:] = parent1[:point], parent2[point:]
		child2[:point], child2[point:] = parent2[:point], parent1[point:]
	return (child1, child2)

def mutation(bitstring, mut_rate):
	for i in range(len(bitstring)):
		if np.random.rand() < mut_rate:
			bitstring[i] = 1 - bitstring[i]

def genetic_algorithm(objective, bounds, Xy, group_bitlens, target, swp_func, invalid_penalty, max_iter, n_bits, n_pop, cross_rate, mut_rate):
	seen_reactions = set()
	pop = [np.random.randint(0, 2, n_bits*len(bounds)) for i in range(n_pop)]
	gen = 0
	while gen < max_iter:
		one_hot_encodings = [decode(p, group_bitlens, n_bits, bounds) for p in pop]
		scores = []
		for one_hot in one_hot_encodings:
			score = objective(one_hot, Xy, target, swp_func, group_bitlens, invalid_penalty)
			scores.append(score)
			reaction_number = onehot_to_int(one_hot)
			swapped_reaction_number = onehot_to_int(swp_func(one_hot, group_bitlens))
			index = ((Xy[:,:-1] == one_hot).all(axis=1) | (Xy[:,:-1] == swp_func(one_hot, group_bitlens)).all(axis=1)).nonzero()[0]
			barrier = Xy[index[0],-1]
			if swapped_reaction_number not in seen_reactions:
				seen_reactions.add(reaction_number)
			if score < 1.0:
				target_number = len(seen_reactions)
				return target_number
		selected = [selection(pop, scores) for i in range(n_pop)]
		children = []
		for i in range(0, n_pop, 2):
			parent1, parent2 = selected[i], selected[i+1]
			for child in crossover(parent1, parent2, cross_rate):
				mutation(child, mut_rate)
				children.append(child)
		pop = children
		gen += 1
	# If algorithm has not found exact solution within max_iter
	target_number = len(seen_reactions)
	return target_number

def run_ga_search(dataset, mkswp, bounds, group_bitlens, swp_func, max_iter, n_pop, cross_rate, mut_rate, n_repeats):
	dataset_file = open(dataset.replace("barriers_groups.csv", "ga_results.txt"), "w")
	dataset_file.write(str(datetime.datetime.now().time()) + "\n")
	dataset_file.write(dataset + "\n")
	dataset_file.write("#" * 20 + "\n")
	dataset_file.flush()
	Xy = generate_data(dataset, mkswp)
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

	n_bits = max([2**(int(np.log2(bnd[1])) + 1) for bnd in bounds])
	for i, target in enumerate(targets):
		for seed in range(1, n_repeats+1):
			np.random.seed(seed)
			target_number = genetic_algorithm(difference_from_target, bounds, Xy, group_bitlens, target, swp_func, 2*np.std(Xy[:,-1]), max_iter, n_bits, n_pop, cross_rate, mut_rate)
			dataset_file.write("target = %f seed = %d target_number = %d" % (target, seed, target_number) + "\n")
			dataset_file.flush()
			target_means[i] += target_number
			target_stds[i] += target_number**2

	target_means /= n_repeats
	target_stds = np.sqrt(np.abs((target_stds / n_repeats) - (target_means)**2))
	dataset_file.write("target " + " ".join(str(tm) for tm in target_means) + "\n")
	dataset_file.write("targetstd: " + " ".join(str(tstd) for tstd in target_stds) + "\n")
	dataset_file.write("#" * 20 + "\n")
	dataset_file.close()


if __name__ == "__main__":
	datasets = ["barriers_groups.csv", "vaska_barriers_groups.csv", "e2_barriers_groups.csv", "sn2_barriers_groups.csv"]
	mkswp_functions = [mkswp_ma, mkswp_vaska, mkswp_e2sn2, mkswp_e2sn2]
	swp_functions = [swp_ma_onehot, swp_vaska_onehot, swp_e2sn2_onehot, swp_e2sn2_onehot]
	bounds = [[[0,4], [0,4], [0,4], [0,7]], [[0,10], [0,2], [0,11], [0,11]], [[0,4], [0,4], [0,4], [0,4], [0,2], [0,3]], [[0,4], [0,4], [0,4], [0,4], [0,2], [0,3]], [[0,4], [0,4], [0,4], [0,4], [0,2], [0,3]], [[0,4], [0,4], [0,4], [0,4], [0,2], [0,3]]]
	group_bitlens = [[5, 5, 5, 8], [11, 3, 12, 12], [5, 5, 5, 5, 3, 4], [5, 5, 5, 5, 3, 4], [5, 5, 5, 5, 3, 4], [5, 5, 5, 5, 3, 4]]
	max_iter = 10000
	n_pop = 20
	cross_rate = 0.9
	mut_rate = 0.05
	n_repeats = 25
	for i in range(len(datasets)):
		p = multiprocessing.Process(target=run_ga_search, args=(datasets[i], mkswp_functions[i], bounds[i], group_bitlens[i], swp_functions[i], max_iter, n_pop, cross_rate, mut_rate, n_repeats))
		p.start()
