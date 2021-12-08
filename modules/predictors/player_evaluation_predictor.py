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
        self.ticks_before_middle = 20
        self.scale = 1e-6

    def save_graphs(self, output_dataset):
        split = output_dataset[output_dataset['split'] == 'test']
        true = (10 ** split['value']).values.astype(int)
        graph = MatPlotLibGraph(self.configs)

        x_ticks = self.get_x_ticks(true)
        for f in [
            self.get_abs_diff,
            self.get_rmse,
            self.get_mean_fraction,
        ]:
            self.plot_one_f(graph, f, x_ticks, split, true)
        graph.plot_hist(true, self.pred_dir)

    def plot_one_f(self, graph, f, x_ticks, split, true):
        _true = np.array(true) / 1e6
        ys, labels = [], []
        for tag, model_name in self.configs.get('models').items():
            model_name_tag = f'{model_name}_{tag}'
            loss_y = []
            for prev_x_point, x_point in zip(x_ticks[:-1], x_ticks[1:]):
                idx = np.where(np.logical_and(prev_x_point < _true, _true < x_point))[0]
                if len(idx) > 0:
                    pred = (10 ** split[model_name_tag]).values.astype(int)[idx] / 1e6
                    loss_y.append(f(_true[idx], pred))
            ys.append(loss_y)
            labels.append(f'{model_name_tag} {f.__name__}')
        graph.plot_lines(x_ticks, ys, labels, self.pred_dir, 'value in millions', f.__name__)

    def get_x_ticks(self, true):
        middle_point = int(self.middle / 1e6)
        x_ticks = []
        x_ticks.extend(list(range(0, middle_point,
                                  int(middle_point / self.ticks_before_middle))))
        _x_step = self.x_step / 1e6
        x_ticks.extend(list(range(middle_point, int(max(true) / 1e6),
                                  int(middle_point / self.ticks_before_middle) * 2)))
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







