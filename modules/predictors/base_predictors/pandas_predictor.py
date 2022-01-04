from typing import AnyStr, Dict
import pandas as pd
import torch
from modules.containers.di_containers import TrainerContainer
from modules.helpers.csv_saver import CSVSaver
from modules.helpers.matplotlibgraph import MatPlotLibGraph
from modules.predictors.base_predictors.base_predictor import BasePredictor
from modules.trainers.default_trainer import metrics_fom_torch
from utils.common import unpickle_obj, transform, mean_dict_values, get_data_loaders
from utils.constants import CLASSIFIERS_DIR
from utils.registry import registry
import numpy as np
import csv
import utils.small_functions as sf


@registry.register_predictor('pandas_predictor')
class PandasPredictor(BasePredictor):
    """ uses all wrappers pointed in prediction config to
        make and save several prediction files

        Compares multi prediction models
     """

    def __init__(self, configs: Dict):
        super().__init__(configs)

    def predict_dataset(self, wrapper) -> Dict:
        splits = {}
        data = CSVSaver().load(self.configs)
        const = self.configs.get('static_columns')
        for split_name in self.split_names:
            split = data[data['split'] == split_name]
            split_y = split.iloc[:, const.get('FINAL_LABEL_INDEX')].values
            split_x = split.iloc[:, len(const):]
            preds = wrapper.get_prediction_probs(split_x)

            splits[split_name] = {f'{split_name}_preds': preds,
                                  f'{split_name}_ys': split_y}
        return splits

    def save_graphs(self, output_dataset: pd.DataFrame):
        """ saves various project specific graphs """
