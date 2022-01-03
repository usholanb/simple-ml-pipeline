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
        for split_name in ['train', 'valid', 'test']:
            split = data[data['split'] == split_name]
            split_y = split.iloc[:, const.get('FINAL_LABEL_INDEX')].values
            split_x = split.iloc[:, len(const):]
            preds = self.predict_split_model(split_x, wrapper,
                                             wrapper.name)

            splits[split_name] = {f'{split_name}_preds': preds,
                                  f'{split_name}_ys': split_y}
        return splits

    def predict_split_model(self, split_x, wrapper, model_name_tag):
        preds = {}
        probs = wrapper.get_prediction_probs(split_x)
        # if self.configs.get('classification'):
        #     if len(wrapper.clf.classes_) > 1:
        #         for i, label in enumerate(wrapper.clf.classes):
        #             pred_name = f'{model_name_tag}_{i}'
        #             preds[pred_name] = probs[:, i]
        #     else:
        #         pred_name = model_name_tag
        #         preds[pred_name] = probs
        # else:
        #     if wrapper.n_outputs > 1:
        #         for i in range(wrapper.n_outputs):
        #             preds[f'{model_name_tag}_{i}'] = probs[:, i]
        #     else:
        #         preds[model_name_tag] = probs[:]
        # preds = pd.DataFrame(preds)
        # preds['k_fold'] = model_name_tag.split(wrapper.name)[1]
        return probs

    def save_graphs(self, output_dataset: pd.DataFrame):
        """ saves various project specific graphs """
