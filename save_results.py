import pickle
import numpy as np
import csv
from collections import defaultdict

# ------------------------------------- For saving all the results

all_results = defaultdict(lambda: defaultdict(list))

for data_name in ['michael', 'vaska', 'e2_mp2', 'sn2_mp2']:

    means = []
    stds = []

    means_tosave = []
    stds_tosave = []
    metrics_tosave = [['', 'Global', 'First/Last 3/5', 'Middle', 'Min', 'Max']]

    if data_name == 'vaska': ind = 3
    else: ind = 5

    for filename, label in zip(['random/{}/results_dict.txt'.format(data_name), 
                                'local_search/{}/results_dict_local.txt'.format(data_name), 
                                'local_search/{}/results_dict_greedy.txt'.format(data_name)], 
                               ['Random Search', 'Local Search', 'Guided Local']):

        results = pickle.load(open(filename, "rb"))

        means.append(np.mean(list(results.values()), axis=1)[np.argsort(list(results.keys()))])
        stds.append(np.std(list(results.values()), axis=1)[np.argsort(list(results.keys()))])
        means_tosave.append(np.append(label, means[-1]))
        stds_tosave.append(np.append(label, stds[-1]))

        metrics_tosave.append([label, *np.round([np.mean(means[-1]), np.mean(np.append(means[-1][:ind], means[-1][-ind:])), np.mean(means[-1][ind:-ind]), means[-1][0], means[-1][-1]], 2)])

        all_results['all_averages'][label].append(round(np.mean(means[-1]), 2))
        all_results['end_averages'][label].append(round(np.mean(np.append(means[-1][:ind], means[-1][-ind:])), 2))
        all_results['mid_averages'][label].append(round(np.mean(means[-1][ind:-ind]), 2))
        all_results['min_averages'][label].append(round(means[-1][0], 2))
        all_results['max_averages'][label].append(round(means[-1][-1], 2))
                
    means_tosave.insert(0, np.append('Target Barrier', np.sort(list(results.keys()))))
    stds_tosave.insert(0, np.append('Target Barrier', np.sort(list(results.keys()))))

    with open('random_local_tables/{}_means.csv'.format(data_name), 'w', newline='') as file: 
        csvwriter = csv.writer(file)
        for row in np.array(means_tosave).T:
            csvwriter.writerow(row)

    with open('random_local_tables/{}_stds.csv'.format(data_name), 'w', newline='') as file: 
        csvwriter = csv.writer(file)
        for row in np.array(stds_tosave).T:
            csvwriter.writerow(row)

    with open('random_local_tables/{}_metrics.csv'.format(data_name), 'w', newline='') as file: 
        csvwriter = csv.writer(file)
        for row in metrics_tosave:
            csvwriter.writerow(row)

# --------------------------------------------------- For printing results of all barriers.

    print('\n', data_name, '\n')

    print('SI')

    means = np.round(means, 2)
    stds = np.round(stds, 2)

    for i, barrier in enumerate(np.sort(list(results.keys()))):

        if min(means[0][i], means[1][i], means[2][i]) == means[0][i]:
            print('{} & \\textbf{{{} $\\pm$ {}}} & {} $\\pm$ {} & {} $\\pm$ {} & & & & \\\\'.format(round(barrier, 2), means[0][i], stds[0][i], means[1][i], stds[1][i], means[2][i], stds[2][i]))
        elif min(means[0][i], means[1][i], means[2][i]) == means[1][i]:
            print('{} & {} $\\pm$ {} & \\textbf{{{} $\\pm$ {}}} & {} $\\pm$ {} & & & & \\\\'.format(round(barrier, 2), means[0][i], stds[0][i], means[1][i], stds[1][i], means[2][i], stds[2][i]))
        elif min(means[0][i], means[1][i], means[2][i]) == means[2][i]:
            print('{} & {} $\\pm$ {} & {} $\\pm$ {} & \\textbf{{{} $\\pm$ {}}} & & & & \\\\'.format(round(barrier, 2), means[0][i], stds[0][i], means[1][i], stds[1][i], means[2][i], stds[2][i]))

    print('\n\n') 

# --------------------------------------------------- For printing metric results layout 1

    print('METRICS')

    for row in np.array(metrics_tosave).T[1:]:

        if row[1:].astype(float).min() == float(row[1]):
            print('{} & \\textbf{{{}}} & {} & {} & & & & \\\\'.format(*row))
        elif row[1:].astype(float).min() == float(row[2]):
            print('{} & {} & \\textbf{{{}}} & {} & & & & \\\\'.format(*row))
        elif row[1:].astype(float).min() == float(row[3]):
            print('{} & {} & {} & \\textbf{{{}}} & & & & \\\\'.format(*row))

    print('\n\n') 

print('\n\n\n\n')  

# --------------------------------------------------- For printing metric results layout 2

for metric, dic in all_results.items():
            
    print('\n', metric, '\n')

    for data_name, row in zip(['MA', 'H$_2$', 'E2', 'S$_N$2'], np.array(list(dic.values())).T):

        if row.min() == row[0]:
            print('{} & \\textbf{{{}}} & {} & {} & & & & \\\\'.format(data_name, *row))
        elif row.min() == row[1]:
            print('{} & {} & \\textbf{{{}}} & {} & & & & \\\\'.format(data_name, *row))
        elif row.min() == row[2]:
            print('{} & {} & {} & \\textbf{{{}}} & & & & \\\\'.format(data_name, *row))

    print('\n\n')