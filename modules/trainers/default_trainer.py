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

    def prepare_train(self, split_i, label_i):
        dataset = self.dataset
        split_column = dataset.iloc[:, split_i]

        data = {}
        for split in ['train', 'valid', 'test']:
            data[f'{split}_y'] = dataset.loc[split_column == split].iloc[:, label_i]
            data[f'{split}_x'] = dataset.loc[split_column == split].iloc[:, 2:]
        return data

