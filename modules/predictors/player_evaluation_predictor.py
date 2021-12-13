import pandas as pd
from modules.helpers.csv_saver import CSVSaver
from modules.helpers.matplotlibgraph import MatPlotLibGraph
from modules.predictors.base_predictors.predictor import Predictor
from utils.common import unpickle_obj
from utils.constants import CLASSIFIERS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR
from copy import deepcopy
from utils.registry import registry
import math
import numpy as np
from sklearn.metrics import mean_squared_error
from itertools import combinations


@registry.register_predictor('player_evaluation')
class PlayerEvaluationPredictor(Predictor):
    """ uses all wrappers pointed in prediction config to
        make and save several prediction files """

    def __init__(self, configs, dataset):
        super().__init__(configs, dataset)
        self.x_step = int(5e6)
        self.y_step = int(2e6)
        self.middle = int(80e6)

        self.scale = 1e-6

    def save_graphs(self, output_dataset):
        graph = MatPlotLibGraph(self.configs)

        x_ticks = self.get_x_ticks()
        for f in [
            self.get_abs_diff,
            self.get_rmse,
            self.get_mean_fraction,
        ]:
            self.plot_one_f(graph, f, x_ticks, output_dataset)
        self.one_hist(graph, x_ticks, output_dataset, self.pred_dir, 'distribution')

        # for feature_name in ['Loan_end_year']:
        #     graph.plot_hist(x_ticks, [split[feature_name].to_list()],
        #                     self.pred_dir, f'{feature_name}_{split_name}')

    def one_hist(self, graph, x_ticks, output_dataset, pred_dir, name):
        labels, trues = [], []
        for split_name in ['train', 'test', 'valid']:
            split = output_dataset[output_dataset['split'] == split_name]
            trues.append((10 ** split['value']).values.astype(int) / 1e6)
            labels.append(f'{name}_{split_name}')
        graph.plot_hist(x_ticks, trues, pred_dir, labels)

    def plot_one_f(self, graph, f, x_ticks, output_dataset):
        ys, labels = [], []
        for tag, model_name in self.configs.get('models').items():
            model_name_tag = f'{model_name}_{tag}'
            for split_name in ['train', 'test', 'valid']:
                split = output_dataset[output_dataset['split'] == split_name]
                pred = (10 ** split[model_name_tag]).values.astype(int) / 1e6
                true = (10 ** split['value']).values.astype(int) / 1e6
                label = f'{model_name_tag} {f.__name__}_{split_name}'
                loss_y = []
                for prev_x_point, x_point in zip(x_ticks[:-1], x_ticks[1:]):
                    idx = np.where(np.logical_and(prev_x_point < true, true < x_point))[0]
                    if len(idx) > 0:
                        loss_y.append(f(true[idx], pred[idx]))
                    else:
                        loss_y.append(0)
                ys.append(loss_y)
                labels.append(label)

        graph.plot_grid(x_ticks[:-1], ys, labels, self.pred_dir, 'value in millions', f'{f.__name__}')

    def get_x_ticks(self):
        middle_point = int(self.middle / 1e6)
        low_middle = 10
        x_ticks = []
        x_ticks.extend(list(range(0, low_middle,
                                  1)))
        x_ticks.extend(list(range(low_middle, middle_point,
                                  int(self.x_step / 1e6))))
        _x_step = self.x_step / 1e6
        x_ticks.extend(list(range(middle_point, 200,
                                  int(self.x_step / 1e6) * 2)))
        return np.array(x_ticks)

    def get_abs_diff(self, t, p):
        """ """
        return np.absolute(t - p).sum() / len(t)

    def get_mean_fraction(self, t, p):
        """ """
        return (np.absolute(t - p) / t).sum() / len(t)

    def get_rmse(self, t, p):
        """ """
        return mean_squared_error(t, p, squared=False)







