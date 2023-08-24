import os
import re
import numpy as np

def mkswp_e2sn2(s):
	return [s[1]] + [s[0]] + [s[3]] + [s[2]] + s[4:]

def mkswp_vaska(s):
	return [s[0]] + [s[1]] + [s[3]] + [s[2]]

def mkswp_ma(s):
	return s

def generate_data(barriers_groups_file, mkswp):
	file = open(barriers_groups_file, "r")
	lines = file.readlines()
	file.close()
	columns = lines[0].rstrip("\n").split(",")[:-1]
	unique_groups = {col: [] for col in columns}
	known_mols = []
	barriers = []
	for i in range(1, len(lines)):
		split_line = lines[i].rstrip("\n").split(",")
		split_line = [re.sub("\(?\[\*:\d\]\)?", "", c) for c in split_line]
		known_mols.append([c for c in split_line[:-1]])
		barriers.append(float(split_line[-1]))
		for j, col in enumerate(unique_groups.keys()):
			if split_line[j] not in unique_groups[col]:
				unique_groups[col].append(split_line[j])
	unique_groups = {col: sorted(unique_groups[col]) for col in unique_groups}
	# Removing symmetry duplicates
	to_remove = []
	for i in range(len(known_mols)):
		if mkswp(known_mols[i]) in known_mols[:i]:
			to_remove.append(i)
	for index in reversed(to_remove):
		known_mols.pop(index)
		barriers.pop(index)
	mols = []
	group_counts = [len(unique_groups[g]) for g in unique_groups.keys()]
	incrs = [0 for col in columns]
	while incrs[0] != group_counts[0]:
		s = []
		for i in range(len(columns)):
			s += [unique_groups[columns[i]][incrs[i]]]
		if s not in mols and mkswp(s) not in mols:
			mols.append(s)
		incrs[-1] += 1
		for i in range(len(incrs)-1, 0, -1):
			if incrs[i] >= group_counts[i] and i >= 0:
				incrs[i] = 0
				incrs[i-1] += 1
	mols = sorted(mols)
	Xy = np.zeros((len(mols), sum(group_counts) + 1))
	group_indices = [0] + list(np.cumsum(group_counts)[:-1])
	group_positions = [{} for g in group_indices]
	for i in range(len(mols)):
		for t in range(len(columns)):
			if mols[i][t] not in group_positions[t]:
				group_positions[t][mols[i][t]] = group_indices[t]
				group_indices[t] += 1
			Xy[i,group_positions[t][mols[i][t]]] = 1
		if mols[i] in known_mols:
			index = known_mols.index(mols[i])
			Xy[i,-1] = barriers[index]
		elif mkswp(mols[i]) in known_mols:
			index = known_mols.index(mkswp(mols[i]))
			Xy[i,-1] = barriers[index]
		else:
			Xy[i,-1] = np.nan
	return Xy
