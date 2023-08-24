# finding_barriers

This repository contains the code and data used for the work entitled "Reformulating Reactivity Design for Data-Efficient Machine Learning". The "random", "local_search", "genetic_algorithm", "feature_barrier", "find_barrier" and "reactant_structures" directories contain, respectively, the files for the random search, local search, genetic algorithm, the ML-based search algorithm, the ML-based search algorithm that without using the low-level barriers as a feature and mol2 files for all of the Michael acceptor structures.

## Python version and libraries

This code was tested using Python version 3.9.6 and the libraries numpy, scikit-learn, pandas, matplotlib and rdkit (versions specified in requirements.txt). However, there should be little difficulty if using slightly different versions of either the dependencies or Python itself.

## random

This folder contains six sub-directories, with each containing the data and results for the barrier data sets of the reactions: "e2_lccsd", "e2_mp2", "michael", "sn2_lccsd", "sn2_mp2" and "vaska". The files that are contained within each of these sub-directories are as follows:

* \*barriers_groups\*.csv: Contains the R-groups for each reaction in the data set, along with the calculated activation barrier in Hartrees (for the Michael-addition dataset) or kcal/mol (for all other datasets).

The main directoy contains three python files:

* run.py: Code that collects results of the random sampling procedure for every dataset, saved in tempory files in each dataset sub-directory.
* convert_results.py: Code that combines all the results in each sub-directory into the results \*.txt files in the sub-directories, then deletes the tempory files.
* plot_paper.py: Code that plots the results figures using all \*.txt files in the sub-directories

## local_search

This folder contains six sub-directories, with each containing the data and results for the barrier data sets of the reactions: "e2_lccsd", "e2_mp2", "michael", "sn2_lccsd", "sn2_mp2" and "vaska". The files that are contained within each of these sub-directories are as follows:

* \*barriers_groups\*.csv: Contains the R-groups for each reaction in the data set, along with the calculated activation barrier in Hartrees (for the Michael-addition dataset) or kcal/mol (for all other datasets).

The main directoy contains six python files:

* Heuristic.py and Heuristic_all.py: Code that defines the direct and multiple local search agents respectively.
* run_targ.py and run_min_max_targ.py: Code that collects results for the direct and multiple local search procedures respectively, for every dataset. The results for the direct and multiple local searches are saved in \*.txt files in the targ and min_max_targ sub-sub-directories respectively, in each dataset sub-directory.
* plot_paper_targ.py and plot_paper_min_max_targ: Code that plots the results figures using all \*.txt files in the sub-sub-directories.

The main directory also contains a flow chart outlining the local search procedure.

## feature_barrier

This directory contains the scripts pertaining to the majority of the model-based results from this work. The purposes of the files in this directory are as follows:

* feature_barrier.py: The script that implements the main ML-based search algorithm using the approximate low-level calculated barriers as an input feature.
* bayes_feature_barrier.py: Implements the Bayesian optimization algorithm using expected improvement values for sampling.
* reaction_repr.py: This script provides functions that generate the input data for the ML models and search algorithms which use the low-level barriers as input features in this work. Generates one-hot encoding representations and computes RDKit descriptors for the Michael addition reactants and Vaska's complex catalyst structures.
* model_evaluation.py: Runs assessment of six scikit-learn ML models trained on 30 data points and using one-hot encoding as an input representation.
* model_evalulation_features.py: Same as model_evaluation.py but using RDKit descriptors as input representations for the aza-Michael addition and dihydrogen activation datasets.
* ml_analysis.py: Script that calculates feature importances and model test and train errors during search procedure.
* scramble_features.py: Script that runs same search procedure as feature_barrier.py but with randomly shuffled input features.
* lab_test.py: Script that runs a simulation of the real world use of our main ML-based search algorithm, starting with a dihydrogen activation reaction with a barrier of 20 kcal/mol and the intention to minimize that barrier as much as possible.
* \*barrier_feature_groups.csv: Contain the R-groups for each reaction in the dataset, along with the calculated activation barriers (low-level and higher level) in kcal/mol.
* vaska_\*_spes.txt: GoodVibes output files for the LDA single-point energy calculations on the catalyst (react) and transition state (ts) structures for the dihydrogen activation dataset.

The \*barrier_feature_groups.csv files contain data from the publication "G. F. von Rudorff, S. N. Heinen, M. Bragato, O. A. von Lilienfeld, Thousands of reactants and transition states for competing E2 and SN2 reactions, Machine Learning: Science and Technology 1, 045026 (2020). doi:10.1088/2632-2153/aba822", available from: "https://archive.materialscloud.org/record/2020.55", as well as data from the publication "P. Friederich, G. dos Passos Gomes, R. De Bin, A. Aspuru-Guzik and D. Balcells, Machine learning dihydrogen activation in the chemical space surrounding Vaska's complex, Chemical Science, 2020, 11, 4584-4601", available from: "https://doi.org/10.5683/SP2/CJS7QA".

## find_target

This directory contains the scripts used for the version of our main ML search algorithm without using the low-level barriers as an input feature. The purposes of the files in this directory are as follows:

* find_target.py: The main script that implements the ML-based search algorithm without using the approximate low-level calculated barriers.
* reaction_repr.py: Same as in the "feature_barrier" directory, but without including the low-level barriers as part of the reaction representations.
* model_evaluation.py: Same as in the "feature_barrier" directory.
* \*barriers_groups.csv: Same as in the "feature_barrier" directory but without low-level barriers included.

## genetic_algorithm

This directory contains the code implementing the genetic algorithm. The purposes of the files in this directory are as follows:

* genetic_algorithm.py: Script running the search using the genetic algorithm.
* reaction_repr.py: Same as in the "find_target" directory.
* \*barriers_groups.csv: Same as in the "find_target" directory.