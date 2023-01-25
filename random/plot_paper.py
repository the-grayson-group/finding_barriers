import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot(file, barrier_range, plot_name, min_max_range):

    colors = ["#113956", "#b25503", "#186618", "#871616", "#49266b", "#135b63"]
    colors = {barrier: ["C{}".format(i), colors[i]] for i, barrier in enumerate(barrier_range)}

    handles = []
    for barrier in barrier_range:
        
        results = np.array(pickle.load(open('{}/{}_dict.txt'.format(file, barrier), "rb")))
        
        dev_means = np.mean(results[:, 2], axis=0)[:480]
        dev_stds = np.std(results[:, 2], axis=0)[:480]

        x_data = np.arange(len(dev_means))
        y_data = dev_means

        handles.append(plt.plot(x_data, y_data, '-', color=colors[barrier][0], label=str(barrier) + ' kcal mol$^{-1}$')[0])
        plt.fill_between(x_data, y_data, y_data + dev_stds, color=colors[barrier][0], alpha=0.3)
        plt.fill_between(x_data, y_data, y_data - dev_stds, color=colors[barrier][0], alpha=0.3)
        plt.plot([12, 490], [barrier, barrier], "--", linewidth=1.0, color=colors[barrier][1])

    plt.xticks(np.arange(24, 457, 48))
    plt.yticks(barrier_range)
    plt.xlabel('Number of samples')
    plt.ylabel('Activation barrier / kcal mol$^{-1}$')
    plt.legend(handles=handles[::-1], loc='upper right')
    plt.title('RS, {}: Found target barriers'.format(plot_name))
    plt.savefig('{}/{}_random_target.png'.format(file, file), dpi=600)
    plt.clf()

    for barrier in barrier_range:
        
        results = np.array(pickle.load(open('{}/{}_dict.txt'.format(file, barrier), "rb")))

        barriers_mins = results[:, 0]
        barriers_max = results[:, 1]
        
        dev_means_min = np.mean(barriers_mins, axis=0)[:480]
        dev_stds_mins = np.std(barriers_mins, axis=0)[:480]
        dev_means_max = np.mean(barriers_max, axis=0)[:480]
        dev_stds_max = np.std(barriers_max, axis=0)[:480]

        x_data = np.arange(len(dev_means))

        plt.plot(x_data, dev_means_min, '-', color=colors[barrier][0], label=str(barrier) + ' kcal mol$^{-1}$')
        plt.fill_between(x_data, dev_means_min, dev_means_min + dev_stds_mins, color=colors[barrier][0], alpha=0.3)
        plt.fill_between(x_data, dev_means_min, dev_means_min - dev_stds_mins, color=colors[barrier][0], alpha=0.3)
        plt.plot(x_data, dev_means_max, '-', color=colors[barrier][0])
        plt.fill_between(x_data, dev_means_max, dev_means_max + dev_stds_max, color=colors[barrier][0], alpha=0.3)
        plt.fill_between(x_data, dev_means_max, dev_means_max - dev_stds_max, color=colors[barrier][0], alpha=0.3)

    plt.plot([12, 490], [barriers_mins.min(), barriers_mins.min()], "--", linewidth=1.0, color="black")
    plt.plot([12, 490], [min_max_range[0], min_max_range[0]], "--", linewidth=1.0, color="grey")
    plt.plot([12, 490], [barriers_max.max(), barriers_max.max()], "--", linewidth=1.0, color="black")
    plt.plot([12, 490], [min_max_range[1], min_max_range[1]], "--", linewidth=1.0, color="grey")

    plt.xticks(np.arange(24, 457, 48))
    plt.yticks(barrier_range)
    plt.xlabel('Number of samples')
    plt.ylabel('Activation barrier / kcal mol$^{-1}$')
    plt.legend()
    plt.title('RS, {}: Found min and max barriers'.format(plot_name))
    plt.savefig('{}/{}_random_minmax.png'.format(file, file), dpi=600)
    plt.clf()

files = ['e2_lccsd', 'e2_mp2', 'michael', 'sn2_lccsd', 'sn2_mp2', 'vaska']
plot_names = ['E2 (uncon, lccsd)', 'E2 (con, mp2)', 'MA', 'Sn2 (uncon, lccsd)', 'Sn2 (con, mp2)', '$H_2$']
barrier_ranges = 2 * [np.arange(-20, 61, 20)] + [np.arange(10, 51, 10)] + 2 * [np.arange(-20, 61, 20)] + [np.arange(0, 26, 5)]
min_max_ranges = [[-15, 45], [-10, 40], [15 ,45], [-10, 60], [-10, 60], [2, 24]]

[plot(file, barrier_range, plot_name, min_max_range) for file, plot_name, barrier_range, min_max_range in zip(files, plot_names, barrier_ranges, min_max_ranges)]


