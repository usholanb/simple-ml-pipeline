import pandas as pd

from modules.helpers.csv_saver import CSVSaver
from modules.predictors.base_predictors.predictor import Predictor
from utils.common import unpickle_obj
from utils.constants import CLASSIFIERS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR
from copy import deepcopy

from utils.registry import registry
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from itertools import combinations


@registry.register_predictor('player_evaluation')
class PlayerEvaluationPredictor(Predictor):
    """ uses all wrappers pointed in prediction config to
        make and save several prediction files """

    def save_graphs(self, output_dataset):
        x_step, y_step = 20e6, 2e6


        split = output_dataset[output_dataset['split'] == 'test']
        true = (10 ** split['value']).values.astype(int)


        def get_uniform_x(true):
            return list(range(0, math.ceil(true.max()), int(x_step)))

        def get_diff(t, p):
            """ """
            return (t - p).sum() / len(t)

        def get_abs_diff(t, p):
            """ """
            return np.absolute(t - p).sum() / len(t)

        def get_rmse(t, p):
            """ """
            return mean_squared_error(t, p, squared=False)

        x_points = get_uniform_x(true)
        for tag, model_name in self.configs.get('models').items():
            model_name_tag = f'{model_name}_{tag}'

            for f in [
                # get_diff,
                # get_abs_diff,
                get_rmse
            ]:
                x, y = [], []
                binsy, binsx = [], []
                for prev_x_point, x_point in zip(x_points[:-1], x_points[1:]):
                    idx = np.where(np.logical_and(prev_x_point < true, true < x_point))[0]
                    b = 0
                    if len(idx) > 0:
                        pred = (10 ** split[model_name_tag]).values.astype(int)[idx]
                        y.append(f(true[idx], pred))
                        x.append(x_point)
                        b += len(idx)
                    binsy.append(b)
                    binsx.append(x_point)

                locs, labels = plt.xticks()
                plt.yticks(np.arange(0, max(y), step=y_step))
                plt.xticks(np.arange(x_points[0], x_points[-1], step=x_step))
                plt.plot(x, y, 'b', label='rmse diff')
                coeff = max(y) / max(binsy)
                plt.plot(binsx, [e * coeff for e in binsy], 'r', label='distribution')
                plt.xlabel('value')
                plt.ylabel(f'{f.__name__} of true prediction {model_name_tag}')
                plt.title(f'{model_name_tag}')
                plt.savefig(f'{self.pred_dir}/{model_name_tag}_{f.__name__}.png')
                plt.clf()








