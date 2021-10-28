import torch

from modules.trainers.base_trainer import BaseTrainer
import sklearn
from typing import AnyStr
import numpy as np

from utils.registry import registry


class DefaultTrainer(BaseTrainer):
    def __init__(self, configs, dataset):
        self.configs = configs
        self.dataset = dataset
        self.label_types = sorted(dataset.iloc[:, self.configs.get('constants').get('FINAL_LABEL_INDEX')].unique())
        self.split_i = self.configs.get('constants').get('FINAL_SPLIT_INDEX')
        self.label_i = self.configs.get('constants').get('FINAL_LABEL_INDEX')
        self.split_column = dataset.iloc[:, self.split_i]

    def prepare_train(self):
        """ splits data to train, test, valid and returns numpy array """
        data = {}
        for split in ['train', 'valid', 'test']:
            data[f'{split}_y'] = self.dataset.loc[self.split_column == split].iloc[:, self.label_i].values
            data[f'{split}_x'] = self.dataset.loc[self.split_column == split].iloc[:, 2:].values
        return data

    def get_wrapper(self):
        wrapper_class = registry.get_wrapper_class(
            self.configs.get('model').get('name'))

        if wrapper_class is not None:
            wrapper = wrapper_class(self.configs, self.label_types)
        else:
            wrapper = registry.get_wrapper_class('special_wrapper')\
                (self.configs, self.label_types)
        return wrapper