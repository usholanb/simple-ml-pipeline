import numpy as np
import torch
from torch.utils.data import Dataset


class DefaultDataset(Dataset):

    def __init__(self, x: np.ndarray, y: np.ndarray, configs, split_name):
        self.configs = configs
        self.batch_size = configs.get('dataset').get('data_loaders', {})\
            .get('split_name', {}).get('batch_size', 32)
        self.x = x
        self.y = y
        self.split_name = split_name
        self.shape_dataset()

    def shape_dataset(self):
        self.x = torch.FloatTensor(self.x)
        self.y = torch.Tensor(self.y).reshape((len(self.x), -1))
        self.y = self.y if self.y.shape[1] > 1 else self.y.flatten()
        if self.classification:
            self.y = self.y.long()

    @property
    def classification(self):
        return self.configs.get('dataset').get('label_type') == 'classification'

    @property
    def regression(self):
        return self.configs.get('dataset').get('label_type') == 'regression'

