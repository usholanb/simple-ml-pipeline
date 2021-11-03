import numpy as np
import pandas as pd
from ray import tune
from modules.trainers.base_trainer import BaseTrainer
from typing import Dict, AnyStr
from utils.common import inside_tune, setup_imports
from modules.wrappers.base_wrappers.base_wrapper import BaseWrapper
from utils.common import pickle_obj
from utils.constants import CLASSIFIERS_DIR
from utils.registry import registry


class DefaultTrainer(BaseTrainer):
    def __init__(self, configs: Dict, dataset: pd.DataFrame):
        self.configs = configs
        self.dataset = dataset
        self.split_i = self.configs.get('static_columns').get('FINAL_SPLIT_INDEX')
        self.label_i = self.configs.get('static_columns').get('FINAL_LABEL_INDEX')
        self.label_types = {v: index for index, v in enumerate(sorted(self.dataset.iloc[:, self.label_i].unique()))}
        self.split_column = dataset.iloc[:, self.split_i]
        self.wrapper = None

    def prepare_train(self) -> Dict:
        """ splits data to train, test, valid and returns numpy array """
        data = {}
        features_list = self.configs.get('features_list')
        if not features_list:
            print('features_list not specified')
        f_list = DefaultTrainer.figure_feature_list(features_list, self.dataset.columns)
        for split in ['train', 'valid', 'test']:
            data[f'{split}_y'] = \
                self.dataset.loc[self.split_column == split].iloc[:, self.label_i].values
            if features_list:
                data[f'{split}_x'] = \
                    self.dataset.loc[self.split_column == split][f_list].values
            else:
                data[f'{split}_x'] = \
                    self.dataset.loc[self.split_column == split].iloc[:, len(self.configs.get('static_columns')):].values
        self.configs['features_list'] = f_list
        return data

    def get_wrapper(self) -> BaseWrapper:
        wrapper_class = registry.get_wrapper_class(
            self.configs.get('model').get('name'))

        if wrapper_class is not None:
            wrapper = wrapper_class(self.configs, self.label_types)
        else:
            wrapper = registry.get_wrapper_class('special_wrapper')\
                (self.configs, self.label_types)
        self.wrapper = wrapper
        return wrapper

    def model_path(self) -> AnyStr:
        return f'{CLASSIFIERS_DIR}/{self.wrapper.name}.pkl'

    def save(self) -> None:
        pickle_obj(self.wrapper, self.model_path())

    def log_metrics(self, results):
        if inside_tune():
            tune.report(**results)
        else:
            to_print = '  '.join([f'{k}: {"{:10.4f}".format(v)}' for k, v in results.items()])
            print(to_print)

    def print_metrics(self, data):

        for split_name in ['train', 'valid', 'test']:
            outputs = self.wrapper.forward(data[f'{split_name}_x'])
            s_metrics = self.get_split_metrics(data[f'{split_name}_y'], outputs)
            s_metrics = "\n".join([f"{k}:{v}" for k, v in s_metrics.items()])
            print(f'{split_name}:\n{s_metrics}\n')

    def get_split_metrics(self, y_true, y_outputs):
        setup_imports()
        metrics = self.configs.get('trainer').get('metrics')
        metrics = metrics if isinstance(metrics, list) else [metrics]
        results = {}
        for metric_name in metrics:
            metric = registry.get_metric_class(metric_name)()
            results[metric_name] = metric.compute_metric(y_true, y_outputs)
        return results

    @staticmethod
    def figure_feature_list(f_list, available_features):
        final_list = []
        for available_feature in available_features:
            for feature in f_list:
                if feature == available_feature \
                        or '_'.join(available_feature.split('_')[:-1]) == feature:
                    final_list.append(available_feature)
        return final_list

