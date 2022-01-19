from typing import AnyStr, Dict, Callable
from modules.helpers.matplotlibgraph import MatPlotLibGraph
from modules.predictors.simple_predictor import SimplePredictor
from utils.registry import registry
import numpy as np
import utils.small_functions as custom_functions_module


@registry.register_predictor('player_valuation_predictor')
class PlayerValuationPredictor(SimplePredictor):
    """ uses all wrappers specified in prediction config to
        create several prediction files """

    def __init__(self, configs: Dict):
        super().__init__(configs)
        self.scale = 1e-6
        self.points = [10, 50, 110]
        self.steps = [1, 5, 10]
        self.graph = MatPlotLibGraph(self.configs)
        self.split_names = self.configs.get('splits', [])

    def save_graphs(self, preds_ys: Dict) -> None:
        x_ticks = self.get_x_ticks()
        for f_name in self.configs.get('plots', {}).get('steps_func', []):
            f = getattr(custom_functions_module, f_name)
            self.plot_one_f(f, x_ticks, preds_ys)
        self.one_hist(x_ticks, preds_ys, 'distribution')

    def one_hist(self, x_ticks: np.ndarray, preds_ys: Dict,
                 name: AnyStr) -> None:
        """ ys are the same for all models """
        labels, trues = [], []
        splits = list(preds_ys.values())[0]
        for split_name in self.split_names:
            split = splits[split_name]
            y_name = f'{split_name}_ys'
            trues.append((10 ** split[y_name]).astype(int) / 1e6)
            labels.append(f'{name}_{split_name}')
        self.graph.plot_hist(x_ticks, trues, self.pred_dir, labels)

    def plot_one_f(self, f: Callable, x_ticks: np.ndarray, preds_ys: Dict) -> None:
        ys, labels, quantities = [], [], []
        for model_name_tag, splits in preds_ys.items():
            for split_name, split in splits.items():
                pred_name = f'{split_name}_preds'
                y_name = f'{split_name}_ys'
                pred = (10 ** split[pred_name]).astype(int) / 1e6
                true = (10 ** split[y_name]).astype(int) / 1e6
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

