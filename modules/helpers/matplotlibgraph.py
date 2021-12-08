import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


class MatPlotLibGraph:

    def __init__(self, configs):
        self.configs = configs

    def plot_lines(self, x, ys, lines_labels, save_folder, x_label, y_label):
        figure(figsize=(50, 30), dpi=100)
        plt.locator_params(axis="y", nbins=10)
        plt.yticks(fontsize=30)
        plt.xticks(x, fontsize=30)
        plt.xlabel(x_label, fontsize=50)
        plt.ylabel(y_label, fontsize=50)
        plt.ticklabel_format(style='plain', useMathText=True)
        for y, line_label in zip(ys, lines_labels):
            plt.plot(x[:-1], y, label=line_label, linewidth=7.0)
            plt.legend(fontsize=40, loc="upper left")
        plt.savefig(f'{save_folder}/{y_label}.png')
        plt.clf()

    def plot_hist(self, true, save_dir):
        plt.xticks(list(range(0, int(max(true) / 1e6), int((max(true) / 1e6) / 20))), fontsize=30)
        plt.yticks(fontsize=30)
        plt.hist(true / 1e6, 50, density=True, facecolor='g')
        plt.xlabel('value in millions', fontsize=50)
        plt.ylabel('quantity of players', fontsize=50)
        plt.savefig(f'{save_dir}/distribution.png')
