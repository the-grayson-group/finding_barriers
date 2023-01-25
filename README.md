# finding_barriers
This repository contains the code and data used for the work entitled "Redefining Machine Learning Activation Energies Towards Identifying Reactions with Desired Barriers Using Small Amounts of Data". The "random", "local_search" and "supervised" directories contain, respectively, the files for the random search, local search and supervised learning algorithms as described in the paper.

## random

## local_search

## supervised
This folder contains three sub-directories "michael", "e2sn2" and "vaska" that each contain the data and code for the barrier datasets of the reactions: aza-Michael addition, E2 and Sn2 and dihydrogen activation on Vaska's complex. The files that are contained within each of these sub-directories are as follows:
* \*barriers_groups\*.csv: Contains the R-groups for each reaction in the dataset, along with the calculated activation barrier in Hartrees (for the Michael-addition dataset) or kcal/mol (for all other datasets).
* distribution_barriers*.py: Code that plots the activation barrier distribution for a dataset.
* \*target_minmax.py: Main code that implements the supervised learning algorithm and runs the experiments changing the target barriers and the amounts of training data sampled by the model.
* \*print_results.py: Code that plots the results figures using the "results.py" file from the "supervised" super-directory.

The "e2sn2" sub-directory also contains "\*data.txt" and "barriers.txt" files that contain the data from the publication "G. F. von Rudorff, S. N. Heinen, M. Bragato, O. A. von Lilienfeld, Thousands of reactants and transition states for competing E2 and SN2 reactions, Machine Learning: Science and Technology 1, 045026 (2020). doi:10.1088/2632-2153/aba822", available from: "https://archive.materialscloud.org/record/2020.55"

The "vaska" sub-directory also contains the "vaska_features_properties_smiles_filenames.csv" file that contains the data from the publication "P. Friederich, G. dos Passos Gomes, R. De Bin, A. Aspuru-Guzik and D. Balcells, Machine learning dihydrogen activation in the chemical space surrounding Vaska's complex, Chemical Science, 2020, 11, 4584-4601", available from: "https://doi.org/10.5683/SP2/CJS7QA".
