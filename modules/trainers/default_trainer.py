import numpy as np
from ray import tune
from modules.trainers.base_trainer import BaseTrainer
from typing import Dict, AnyStr
from utils.common import inside_tune, setup_imports
from modules.wrappers.base_wrappers.base_wrapper import BaseWrapper
from utils.common import pickle_obj
from utils.constants import CLASSIFIERS_DIR
from utils.registry import registry


class DefaultTrainer(BaseTrainer):
    def __init__(self, configs, dataset):
        self.configs = configs
        self.dataset = dataset
        self.split_i = self.configs.get('constants').get('FINAL_SPLIT_INDEX')
        self.label_i = self.configs.get('constants').get('FINAL_LABEL_INDEX')
        self.label_types = {v: index for index, v in enumerate(sorted(dataset.iloc[:, self.label_i].unique()))}
        self.dataset.iloc[:, self.label_i] = np.array(
            [self.label_types[y] for y in self.dataset.iloc[:, self.label_i].tolist()]
        )
        self.split_column = dataset.iloc[:, self.split_i]
        self.wrapper = None

    def prepare_train(self) -> Dict:
        """ splits data to train, test, valid and returns numpy array """
        data = {}
        f_list = self.configs.get('features_list')
        if not f_list:
            print('features_list not specified')
        for split in ['train', 'valid', 'test']:
            data[f'{split}_y'] = self.dataset.loc[self.split_column == split].iloc[:, self.label_i].values
            if f_list:
                data[f'{split}_x'] = self.dataset.loc[self.split_column == split][f_list].values
            else:
                data[f'{split}_x'] = self.dataset.loc[self.split_column == split].iloc[:, 2:].values
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
        if self.configs.get('trainer').get('save'):
            pickle_obj(self.wrapper, self.model_path())

    def log_metrics(self, results, split_name=''):
        if split_name:
            split_name = f'{split_name}_'
            results = dict([(f'{split_name}{k}', v) for k, v in results.items()])
        if inside_tune():
            tune.report(**results)
        else:
            to_print = '_'.join([f'{k}: {v}' for k, v in results.items()])
            print(to_print)

    def get_metrics(self, data):
        s_metrics = {}
        for split_name in ['train', 'valid', 'test']:
            outputs = self.wrapper.forward(data[f'{split_name}_x'])
            s_metrics[split_name] = self.get_split_metrics(data[f'{split_name}_y'], outputs)
        return s_metrics

    def get_split_metrics(self, y_true, y_outputs):
        setup_imports()
        metrics = {}
        for metric_name in ['accuracy']:
            metric = registry.get_metric_class(metric_name)()
            metrics[metric_name] = metric.compute_metric(y_true, y_outputs)
        return metrics
