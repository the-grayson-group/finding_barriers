import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors


def mkswp_e2sn2(s):
	return [s[1]] + [s[0]] + [s[3]] + [s[2]] + s[4:]

def mkswp_vaska(s):
	return [s[0]] + [s[1]] + [s[3]] + [s[2]]

def mkswp_ma(s):
	return s

def fit_model(model, train_set, parameter_ranges):
	grid_search = GridSearchCV(model, parameter_ranges, scoring="neg_mean_absolute_error", cv=5, refit=True, n_jobs=2)
	grid_search.fit(train_set[:,:-1], train_set[:,-1])
	return grid_search.best_estimator_

def impute_low_barriers(Xy, non_missing_indices, missing_indices):
	model = Ridge()
	parameter_ranges = {"alpha": [10**i for i in range(-7,3)]}
	model = fit_model(model, Xy[non_missing_indices,:-1], parameter_ranges)
	imputed_barriers = model.predict(Xy[missing_indices,:-2])
	return imputed_barriers

def generate_data(barriers_groups_file, mkswp):
	file = open(barriers_groups_file, "r")
	lines = file.readlines()
	file.close()
	columns = lines[0].rstrip("\n").split(",")[:-2]
	unique_groups = {col: [] for col in columns}
	known_mols = []
	low_barriers = []
	barriers = []
	for i in range(1, len(lines)):
		split_line = lines[i].rstrip("\n").split(",")
		split_line = [re.sub("\(?\[\*:\d\]\)?", "", c) for c in split_line]
		known_mols.append([c for c in split_line[:-2]])
		low_barriers.append(float(split_line[-2]))
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
	Xy = np.zeros((len(mols), sum(group_counts) + 2))
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
			Xy[i,-2] = low_barriers[index]
			Xy[i,-1] = barriers[index]
		elif mkswp(mols[i]) in known_mols:
			index = known_mols.index(mkswp(mols[i]))
			Xy[i,-2] = low_barriers[index]
			Xy[i,-1] = barriers[index]
		else:
			Xy[i,-2] = np.nan
			Xy[i,-1] = np.nan
	missing_indices = np.where(np.isnan(Xy[:,-2]))[0]
	non_missing_indices = np.where(~np.isnan(Xy[:,-2]))[0]
	Xy[non_missing_indices,-2] = (Xy[non_missing_indices,-2] - np.mean(Xy[non_missing_indices,-2])) / np.std(Xy[non_missing_indices,-2])
	if missing_indices.size != 0:
		imputed_barriers = impute_low_barriers(Xy, non_missing_indices, missing_indices)
		Xy[missing_indices,-2] = imputed_barriers
	return Xy

def get_one_hot_encoding(barriers_groups_file):
	data = pd.read_csv(barriers_groups_file)
	scaled_low_barriers = (data["low_barrier"] - np.mean(data["low_barrier"])) / np.std(data["low_barrier"])
	group_titles = []
	group_bitlens = []
	for col in data.columns[:-2]:
		group_titles.append(col)
		group_bitlens.append(data[col].nunique())
	group_indices = [0] + list(np.cumsum(group_bitlens)[:-1])
	group_positions = [{} for g in group_indices]
	Xy = np.zeros((len(data), sum(group_bitlens) + 2))
	for i in range(len(data)):
		for t in range(len(group_titles)):
			if data[group_titles[t]][i] not in group_positions[t]:
				group_positions[t][data[group_titles[t]][i]] = group_indices[t]
				group_indices[t] += 1
			Xy[i,group_positions[t][data[group_titles[t]][i]]] = 1
		Xy[i,-2] = scaled_low_barriers[i]
		Xy[i,-1] = data["barrier"][i]
	return Xy

def get_michael_features(csv_file, structure_dir):
	file_regex = re.compile("GS-(\d{1,4})-\d\.mol2")
	file_list = os.listdir(structure_dir)
	file_list.sort(key=lambda file: int(file_regex.search(file).group(1)))
	mols = []
	for file in file_list:
		mol = Chem.MolFromMol2File(structure_dir + "/" + file)
		mols.append(mol)
	descriptors = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
	data = pd.DataFrame(descriptors)
	core_atoms = [0, 1, 2, 3]
	charge_titles = ["Gasteiger_charge_C1", "Gasteiger_charge_C2", "Gasteiger_charge_O1", "Gasteiger_charge_C3"]
	charges = np.zeros((len(mols), len(core_atoms)))
	for m in range(len(mols)):
		for i in range(len(core_atoms)):
			charges[m][i] = mols[m].GetAtomWithIdx(core_atoms[i]).GetDoubleProp("_GasteigerCharge")
	for i in range(len(charge_titles)):
		data[charge_titles[i]] = charges[:,i]
	to_remove = []
	to_scale = []
	for col in data.columns:
		if (data[col] == data[col][0]).all() or col == "Ipc":
			to_remove.append(str(col))
		if ("Count" not in col and "Num" not in col and "fr_" not in col) and col not in to_remove:
			to_scale.append(str(col))
	data = data.drop(to_remove, axis=1)
	column_transformer = ColumnTransformer([("standard_scaler", StandardScaler(), to_scale)], remainder="passthrough")
	transformed_array = column_transformer.fit_transform(data)
	for i, col in enumerate(to_scale):
		data[col] = transformed_array[:,i]
	reaction_info = pd.read_csv(csv_file)
	scaled_low_barriers = (reaction_info["low_barrier"] - np.mean(reaction_info["low_barrier"])) / np.std(reaction_info["low_barrier"])
	data["low_barrier"] = scaled_low_barriers
	data["barrier"] = reaction_info["barrier"]
	Xy = data.to_numpy()
	return Xy

def get_vaska_features(csv_file):
	complex_energies = []
	complex_names = []
	with open("vaska_react_spes.txt", "r") as file:
		for line in file:
			if line.startswith("x  ir"):
				split_line = line.split(" ")
				split_line = list(filter(lambda a: a != "", split_line))
				name = split_line[1].rstrip("_min")
				complex_names.append(name)
				energy = float(split_line[2])
				complex_energies.append(energy)

	ts_energies = []
	ts_names = []
	with open("vaska_ts_spes.txt", "r") as file:
		for line in file:
			if line.startswith("x  ir"):
				split_line = line.split(" ")
				split_line = list(filter(lambda a: a != "", split_line))
				name = split_line[1].rstrip("_ts")
				ts_names.append(name)
				energy = float(split_line[2])
				ts_energies.append(energy)

	lda_barriers = []
	lda_names = []
	h2_energy = -1.168144
	for i in range(len(complex_energies)):
		if complex_names[i] in ts_names:
			index = ts_names.index(complex_names[i])
			lda_barriers.append(627.503 * (ts_energies[index] - (complex_energies[i] + h2_energy)))
			lda_names.append(complex_names[i])

	smiles_df = pd.read_csv(csv_file)
	smiles_strings = smiles_df["smiles"]
	filenames = smiles_df["filename"]
	mols = []
	for smiles_string in smiles_strings:
		mol = Chem.MolFromSmiles(smiles_string)
		mols.append(mol)
	descriptors = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
	data = pd.DataFrame(descriptors)
	core_atom = 0
	charge_title = "Gasteiger_charge_Ir"
	charges = np.zeros((len(mols),))
	for m in range(len(mols)):
		charges[m] = mols[m].GetAtomWithIdx(core_atom).GetDoubleProp("_GasteigerCharge")
	data[charge_title] = charges
	to_remove = []
	to_scale = []
	for col in data.columns:
		if (data[col] == data[col][0]).all() or col == "Ipc" or np.isnan(data[col]).any():
			to_remove.append(str(col))
		if ("Count" not in col and "Num" not in col and "fr_" not in col) and col not in to_remove:
			to_scale.append(str(col))
	data = data.drop(to_remove, axis=1)
	column_transformer = ColumnTransformer([("standard_scaler", StandardScaler(), to_scale)], remainder="passthrough")
	transformed_array = column_transformer.fit_transform(data)
	for i, col in enumerate(to_scale):
		data[col] = transformed_array[:,i]
	low_barriers = []
	for filename in filenames:
		index = lda_names.index(filename)
		low_barriers.append(lda_barriers[index])
	scaled_barriers = (low_barriers - np.mean(low_barriers)) / np.std(low_barriers)
	data["low_barrier"] = scaled_barriers
	data["barrier"] = smiles_df["barrier"]
	Xy = data.to_numpy()
	return Xy
