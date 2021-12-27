import numpy as np
import pandas as pd
from ray import tune

from modules.containers.di_containers import TrainerContainer
from modules.helpers.csv_saver import CSVSaver
from modules.helpers.namer import Namer
from modules.trainers.base_trainer import BaseTrainer
from typing import Dict, AnyStr, List
from utils.common import inside_tune, setup_imports, is_outside_library
from modules.wrappers.base_wrappers.base_wrapper import BaseWrapper
from utils.common import pickle_obj
from utils.constants import CLASSIFIERS_DIR
from utils.registry import registry


class DefaultTrainer(BaseTrainer):
    def __init__(self, configs: Dict):
        self.configs = configs
        self.split_i = self.configs.get('static_columns').get('FINAL_SPLIT_INDEX')
        self.label_index_i = self.configs.get('static_columns').get('FINAL_LABEL_INDEX')
        self.label_i = self.configs.get('static_columns').get('FINAL_LABEL_NAME_INDEX')
        self.classification = self.configs.get('trainer').get('label_type') == 'classification'
        self.wrapper = None
        self.device = TrainerContainer.device

    def prepare_train(self, dataset) -> Dict:
        """ splits data to train, test, valid and returns numpy array """
        data = {}
        features_list = self.configs.get('features_list', [])
        if not features_list:
            print('features_list not specified')
        f_list = DefaultTrainer.figure_feature_list(features_list, dataset.columns.tolist())
        for split in ['train', 'valid', 'test']:
            split_column = dataset.iloc[:, self.split_i]
            data[f'{split}_y'] = \
                dataset.loc[split_column == split].iloc[:, self.label_index_i]
            if features_list:
                data[f'{split}_x'] = \
                    dataset.loc[split_column == split][f_list]
            else:
                data[f'{split}_x'] = \
                    dataset.loc[split_column == split].iloc[:, len(self.configs.get('static_columns')):]
        self.configs['features_list'] = f_list
        return data

    def model_path(self) -> AnyStr:
        return f'{CLASSIFIERS_DIR}/{self.wrapper.name}.pkl'

    def save(self) -> None:
        print(f'saved model {self.model_path}')
        pickle_obj(self.wrapper, self.model_path())

    def _log_metrics(self, results) -> None:
        if inside_tune():
            tune.report(**results)
        else:
            to_print = '  '.join([f'{k}: {"{:10.4f}".format(v)}' for k, v in results.items()])
            print(to_print)

    def print_metrics(self, data: Dict) -> None:
        for split_name in ['train', 'valid', 'test']:
            split_preds = self.wrapper.get_train_probs(data[f'{split_name}_x'])
            s_metrics = self.get_split_metrics(data[f'{split_name}_y'].values, split_preds)
            print(f'{s_metrics}\n')

    def metrics_to_log_dict(self, y_true: np.ndarray, y_preds: np.ndarray, split_name: AnyStr) -> Dict:
        metrics = self.get_split_metrics(y_true, y_preds)
        return dict([(f'{split_name}_{k}', v) for k, v in metrics.items()])

    def get_split_metrics(self, y_true: np.ndarray, y_outputs: np.ndarray) -> Dict:
        setup_imports()
        metrics = self.configs.get('trainer').get('metrics', [])
        metrics = metrics if isinstance(metrics, list) else [metrics]
        results = {}
        for metric_name in metrics:
            metric = registry.get_metric_class(metric_name)()
            results[metric_name] = metric.compute_metric(y_true, y_outputs)
        return results

    @staticmethod
    def figure_feature_list(f_list: List, available_features: List) -> List:
        """ Specifying only name of the feature (w/o index for one hot encoded
            features) is enough to allow them in dataset
            f_list: specified in train config file for training,
            available_features: actually inside dataset
            Return: intersection of two lists"""
        final_list = []
        for available_feature in available_features:
            for feature in f_list:
                if feature == available_feature \
                        or '_'.join(available_feature.split('_')[:-1]) == feature:
                    final_list.append(available_feature)
        return final_list

    def get_dataset(self):
        setup_imports()
        dataset = registry.get_dataset_class(
            self.configs.get('dataset').get('name'))(self.configs)
        return dataset

