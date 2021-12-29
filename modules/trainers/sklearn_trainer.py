import sys
import numpy as np

from modules.helpers.csv_saver import CSVSaver
from modules.trainers.default_trainer import DefaultTrainer, get_metrics
from modules.wrappers.sklearn_wrapper import SKLearnWrapper
from utils.common import setup_imports, is_outside_library, prepare_train, log_metrics
from utils.registry import registry


@registry.register_trainer('sklearn_trainer')
class SKLearnTrainer(DefaultTrainer):

    def __init__(self, configs):
        super().__init__(configs)
        dataset = CSVSaver().load(configs)
        self.data = prepare_train(configs, dataset)
        self.label_name = dataset.columns[self.configs.get('static_columns')
            .get('FINAL_LABEL_INDEX')]
        self.split_column = dataset.iloc[:, self.split_i]

    def train(self) -> None:
        """ trains sklearn model with dataset """
        setup_imports()
        wrapper = self._get_wrapper()
        wrapper.fit(self.data['train_x'], self.data['train_y'])
        valid_pred = wrapper.get_train_probs(self.data['valid_x'])
        train_pred = wrapper.get_train_probs(self.data['train_x'])
        valid_metrics = get_metrics(self.data['valid_y'].values,
                                         valid_pred, 'valid', self.configs)
        train_metrics = get_metrics(self.data['train_y'].values,
                                         train_pred, 'train', self.configs)
        log_metrics({**valid_metrics, **train_metrics})

    def _get_wrapper(self, *args, **kwargs) -> SKLearnWrapper:
        name = self.configs.get('model').get('name')
        if is_outside_library(name):
            self.wrapper = registry.get_wrapper_class('sklearn')\
                (self.configs)
        else:
            raise ValueError(f'{name} doesnt exist in sklearn')
        return self.wrapper









