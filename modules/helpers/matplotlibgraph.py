from typing import Dict
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


class MatPlotLibGraph:

    def __init__(self, configs: Dict):
        self.configs = configs

    def plot_step(self, x, ys, quantities, lines_labels, save_folder, x_label, y_label):
        figure(figsize=(50, 30), dpi=100)
        plt.locator_params(axis="y", nbins=10)
        plt.yticks(fontsize=30)
        plt.xticks(x, fontsize=30)
        plt.xlabel(x_label, fontsize=50)
        plt.ylabel(y_label, fontsize=50)
        plt.ticklabel_format(style='plain', useMathText=True)
        for y, quantity, line_label in zip(ys, quantities, lines_labels):
            plt.step(x, y, where='post', label=line_label, linewidth=7.0)
            plt.legend(fontsize=40, loc="upper left")
            for xx, yy, qq in zip(x, y, quantity):
                plt.text(xx, yy, qq, color='blue', fontweight='bold', fontsize=10)
        plt.savefig(f'{save_folder}/{y_label}.png')
        plt.clf()

    def plot_lines(self, x, ys, lines_labels, save_folder, x_label, y_label):
        figure(figsize=(50, 30), dpi=100)
        plt.locator_params(axis="y", nbins=10)
        plt.yticks(fontsize=30)
        plt.xticks(x, fontsize=30)
        plt.xlabel(x_label, fontsize=50)
        plt.ylabel(y_label, fontsize=50)
        plt.ticklabel_format(style='plain', useMathText=True)
        for y, line_label in zip(ys, lines_labels):
            plt.plot(x, y, label=line_label, linewidth=7.0)
            plt.legend(fontsize=40, loc="upper left")
        plt.savefig(f'{save_folder}/{y_label}.png')
        plt.clf()

    def plot_hist(self, x_ticks, trues, save_dir, labels):
        plt.clf()

        figure(figsize=(50, 30))
        ax = plt.subplot(111)
        colors = ['b', 'r', 'y']
        width = 0.9
        x_ticks = np.array(x_ticks)
        x_ticks_list = [x_ticks - width, x_ticks, x_ticks + width]
        for true, label, c, _x_ticks in zip(trues, labels, colors, x_ticks_list):
            plt.xlabel('value', fontsize=50)
            plt.ylabel('quantity', fontsize=50)
            y_ticks = []
            for prev_x_point, x_point in zip(_x_ticks[:-1], _x_ticks[1:]):
                idx = np.where(np.logical_and(prev_x_point < true, true <= x_point))[0]
                if len(idx) > 0:
                    y_ticks.append(len(idx))
                else:
                    y_ticks.append(0)

            ax.tick_params(axis='both', which='major', labelsize=30)
            ax.tick_params(axis='both', which='minor', labelsize=8)

            ax.bar(_x_ticks[:-1], y_ticks, align='center', width=width,
                    color=c, label=f'{label} Total number : {len(true)}')
            ax.legend(fontsize=40, loc="upper left")

        plt.title(f'distribution')
        plt.savefig(f'{save_dir}/distribution.png')


