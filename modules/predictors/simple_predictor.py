from typing import AnyStr, Dict
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from modules.containers.di_containers import TrainerContainer
from modules.helpers.csv_saver import CSVSaver
from modules.predictors.base_predictors.base_predictor import BasePredictor
from modules.wrappers.base_wrappers.base_wrapper import BaseWrapper
from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper
from utils.common import unpickle_obj, create_folder
from utils.constants import CLASSIFIERS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR
from copy import deepcopy

from utils.registry import registry


@registry.register_predictor('simple_predictor')
class SimplePredictor(BasePredictor):
    """ uses all wrappers pointed in prediction config to
        make and save metrics """

    def __init__(self, configs: Dict):
        self.configs = configs
        self.device = TrainerContainer.device
        self.train_loader, self.valid_loader, self.test_loader = \
            [None] * 3
        self.split_names = self.configs.get('splits', [])

    @property
    def pred_dir(self) -> AnyStr:
        dataset_path = self.configs.get('dataset').get('input_path')
        if dataset_path is not None:
            dataset_name = dataset_path.split('/')[1]
        else:
            dataset_name = self.configs.get('dataset').get('name')
        return create_folder(f'{PREDICTIONS_DIR}/{dataset_name}')

    def get_preds_ys(self) -> Dict:
        model_results = {}
        for tag, model_name in self.configs.get('models').items():
            k_fold_tag = self.configs.get('dataset').get('k_fold_tag', '')
            model_name_tag = f'{model_name}_{tag}{k_fold_tag}'
            model_path = f'{CLASSIFIERS_DIR}/{model_name_tag}.pkl'
            wrapper = unpickle_obj(model_path)
            wrapper.device = self.device
            model_results[model_name_tag] = \
                wrapper.predict_dataset(self.configs, self.split_names)
        return model_results

    def get_metrics(self, split: Dict, split_name: AnyStr) -> Dict:
        """ Saves metrics for the split  """
        y_true = split[f'{split_name}_ys']
        metrics_values = {}
        for metric_name in self.configs.get('metrics', []):
            metric = registry.get_metric_class(metric_name)()
            y_outputs = split[f'{split_name}_preds']
            values = metric.compute_metric(y_true, y_outputs)
            metrics_values[metric_name] = values
        return metrics_values

    def save_metrics(self, preds_ys: Dict) -> None:
        """ """
        metrics = {}
        for model_name, splits in preds_ys.items():
            metrics[model_name] = {}
            for split_name, split in splits.items():
                metrics[model_name][split_name] = self.get_metrics(split, split_name)
        df = pd.concat({k: pd.DataFrame(v) for k, v in metrics.items()})
        CSVSaver.save_file(f'{self.pred_dir}/metrics',
                           df, gzip=False, index=True)

    def yield_model_path(self):
        for tag, model_name in self.configs.get('models').items():
            k_fold_tag = self.configs.get('dataset').get('k_fold_tag', '')
            model_name_tag = f'{model_name}_{tag}{k_fold_tag}'
            model_path = f'{CLASSIFIERS_DIR}/{model_name_tag}.pkl'
            yield model_path, model_name_tag

    def get_split_with_pred(self, preds_ys):
        def insert_into_df(df, i, name, column):
            df.insert(i, name, column, False)
            return i + 1

        data = CSVSaver().load(self.configs)
        for split_name in self.configs.get('splits', []):
            sc_i = len(self.configs.get('static_columns'))
            split = data[data['split'] == split_name]
            model_name_tags = []
            for model_path, model_name_tag in self.yield_model_path():
                preds = preds_ys[model_name_tag][split_name][f'{split_name}_preds']
                sc_i = insert_into_df(split, sc_i, model_name_tag, preds)
                ys = preds_ys[model_name_tag][split_name][f'{split_name}_ys']
                t = 10 ** ys
                p = 10 ** preds
                percentage_diff = np.stack((t / p, p / t), axis=1).max(axis=1).round(2)
                sc_i = insert_into_df(split, sc_i, f'{model_name_tag}_percentage_diff', percentage_diff)
                model_name_tags.append(model_name_tag)
            yield split, split_name, model_name_tags

    def get_prediction_name(self, split_name):
        tag = self.configs.get('dataset').get('tag', '')
        tag = '' if not tag else f'_{tag}'
        return f'{self.pred_dir}/predictions_{split_name}{tag}'

    def save_predictions(self, preds_ys):
        for split, split_name, model_name_tags in self.get_split_with_pred(preds_ys):
            CSVSaver.save_file(self.get_prediction_name(split_name), split, gzip=True, index=False)

    def save_graphs(self, output_dataset: Dict) -> None:
        """ override if need graphs """

