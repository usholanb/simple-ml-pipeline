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

    def predict_dataset(self) -> pd.DataFrame:
        output_dataset = []
        k_fold_tag = self.configs.get('dataset').get('k_fold_tag', '')
        data = CSVSaver().load(self.configs)
        const = self.configs.get('static_columns')
        for tag, model_name in self.configs.get('models').items():
            for split_name in ['train', 'valid', 'test']:
                model_name_tag = f'{model_name}_{tag}'
                model_path = f'{CLASSIFIERS_DIR}/{model_name_tag}{k_fold_tag}.pkl'
                wrapper = unpickle_obj(model_path)
                split = data[data['split'] == split_name]
                split_y = split.iloc[:, const.get('FINAL_LABEL_INDEX')]
                split_x = split.iloc[:, len(const):]
                pred = self.predict_split_model(split_x, wrapper,
                                                model_name_tag, k_fold_tag)
                output_dataset.append(pd.concat([pred, split_y], axis=1))
        return pd.concat(output_dataset, axis=1)
