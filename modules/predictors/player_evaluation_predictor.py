from copy import deepcopy
from typing import AnyStr, Dict
import pandas as pd

from modules.helpers.csv_saver import CSVSaver
from modules.helpers.matplotlibgraph import MatPlotLibGraph
from modules.predictors.base_predictors.base_predictor import BasePredictor
from utils.common import unpickle_obj
from utils.constants import CLASSIFIERS_DIR
from utils.registry import registry
import numpy as np
import utils.small_functions as sf


@registry.register_predictor('player_evaluation_predictor')
class PlayerEvaluationPredictor(BasePredictor):
    """ uses all wrappers pointed in prediction config to
        make and save several prediction files """

    def __init__(self, configs: Dict):
        super().__init__(configs)
        self.scale = 1e-6
        self.points = [10, 80, 200]
        self.steps = [1, 5, 10]
        self.graph = MatPlotLibGraph(self.configs)
        self.split_names = self.configs.get('plots', {}).get('splits', [])

    def save_graphs(self, output_dataset: pd.DataFrame) -> None:
        x_ticks = self.get_x_ticks()
        for f_name in self.configs.get('plots', {}).get('steps_func', []):
            f = getattr(sf, f_name)
            self.plot_one_f(f, x_ticks, output_dataset)
        self.one_hist(x_ticks, output_dataset, 'distribution')

    def one_hist(self, x_ticks: np.ndarray, output_dataset: pd.DataFrame,
                 name: AnyStr) -> None:
        labels, trues = [], []
        for split_name in self.split_names:
            split = output_dataset[output_dataset['split'] == split_name]
            trues.append((10 ** split['value']).values.astype(int) / 1e6)
            labels.append(f'{name}_{split_name}')
        self.graph.plot_hist(x_ticks, trues, self.pred_dir, labels)

    def plot_one_f(self, f, x_ticks, output_dataset):
        ys, labels, quantities = [], [], []
        for tag, model_name in self.configs.get('models').items():
            model_name_tag = f'{model_name}_{tag}'
            for split_name in self.split_names:
                split = output_dataset[output_dataset['split'] == split_name]
                pred = (10 ** split[model_name_tag]).values.astype(int) / 1e6
                true = (10 ** split['value']).values.astype(int) / 1e6
                label = f'{model_name_tag} {f.__name__}_{split_name}'
                loss_y, quantity = [], []
                for prev_x_point, x_point in zip(x_ticks[:-1], x_ticks[1:]):
                    idx = np.where(np.logical_and(prev_x_point < true, true <= x_point))[0]
                    if len(idx) > 0:
                        loss_y.append(f(true[idx], pred[idx]))
                    else:
                        loss_y.append(0)
                    quantity.append(str(len(idx)))
                ys.append(loss_y)
                labels.append(label)
                quantities.append(quantity)
        self.graph.plot_step(x_ticks[:-1], ys, quantities, labels,
                             self.pred_dir, 'value in millions', f'{f.__name__}')

    def get_x_ticks(self) -> np.ndarray:

        x_ticks = []
        current = 0
        for p, s in zip(self.points, self.steps):
            x_ticks.extend(list(range(current, p,
                                      s)))
            current = p
        return np.array(x_ticks)

    def predict(self):
        output_dataset = deepcopy(self.dataset)
        k_fold_tag = self.configs.get('dataset').get('k_fold_tag', '')
        for tag, model_name in self.configs.get('models').items():
            model_name_tag = f'{model_name}_{tag}'
            model_path = f'{CLASSIFIERS_DIR}/{model_name_tag}{k_fold_tag}.pkl'
            wrapper = unpickle_obj(model_path)
            self.print_important_features(wrapper)
            probs = wrapper.predict_proba(self.dataset)
            output_dataset[f'{model_name_tag}'] = probs
            output_dataset['k_fold'] = k_fold_tag
        return output_dataset







