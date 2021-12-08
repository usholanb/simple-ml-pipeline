import pandas as pd
from matplotlib.pyplot import figure
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
        x_step, y_step = 5e6, 2e6

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

        def get_x_ticks(x_points, x_step):
            up = 10
            x_ticks = []
            x_ticks.extend(list(range(int(x_points[0] / 1e6), up, 2)))
            _x_step = x_step / 1e6
            increment_dist = 1.2
            # while up < 200:
            #     _x_step *= increment_dist
            #     up += _x_step / 1e6
            #     x_ticks.append(up)
            x_ticks.extend(list(range(up, 200, 5)))
            return np.array(x_ticks)

        figure(figsize=(50, 30), dpi=2)
        x_points = get_uniform_x(true)
        max_coeff = 0
        for f in [
            # get_diff,
            # get_abs_diff,
            get_rmse
        ]:
            for tag, model_name in self.configs.get('models').items():
                model_name_tag = f'{model_name}_{tag}'
                x, y = [], []
                for prev_x_point, x_point in zip(x_points[:-1], x_points[1:]):
                    idx = np.where(np.logical_and(prev_x_point < true, true < x_point))[0]
                    b = 0
                    if len(idx) > 0:
                        pred = (10 ** split[model_name_tag]).values.astype(int)[idx]
                        y.append(f(true[idx], pred))
                        x.append((prev_x_point + x_point) / 2.)
                        b += len(idx)

                plt.yticks(np.arange(0, max(y) / 1e6, step=y_step / 1e6))

                plt.xticks(get_x_ticks(x_points, x_step))
                plt.ticklabel_format(style='plain', useMathText=True)
                x, y = [[e / 1e6 for e in l] for l in [x, y]]
                plt.plot(x, y, label=f'{model_name_tag} {f.__name__}')
                max_coeff = max(max(y), max_coeff)
                plt.legend()
                plt.xlabel('value')
                plt.ylabel(f'{f.__name__} true VS {model_name_tag}')
                plt.title(f'{model_name_tag}')

                # plt.clf()

            binsy, binsx = [], []
            for prev_x_point, x_point in zip(x_points[:-1], x_points[1:]):
                idx = np.where(np.logical_and(prev_x_point < true, true < x_point))[0]
                b = 0
                if len(idx) > 0:
                    b += len(idx)
                binsy.append(b)
                binsx.append((prev_x_point + x_point) / 2.)
            binsx, binsy = [[e / 1e6 for e in l] for l in [binsx, binsy]]
            plt.plot(binsx, [e * max_coeff / max(binsy) for e in binsy], 'r', label='distribution')
            plt.legend()
            plt.savefig(f'{self.pred_dir}/{f.__name__}.png')








