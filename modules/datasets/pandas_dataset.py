import numpy as np
from modules.datasets.base_datasets.default_dataset import DefaultDataset
from utils.registry import registry


@registry.register_dataset('pandas_dataset')
class PandasDataset(DefaultDataset):

    def __init__(self, x, y, configs, split_name):
        super(PandasDataset, self).__init__(x, y, configs, split_name)
        self.x = self.x.values
        self.y = self.y.values

    def __getitem__(self, index):
        end_index = index + self.batch_size
        sample_x = self.x[index: index + self.batch_size]
        sample_x = np.concatenate([sample_x, self.x[:max(end_index - len(self), 0)]])
        sample_y = self.y[index: index + self.batch_size]
        sample_y = np.concatenate([sample_y, self.y[:max(end_index - len(self), 0)]])
        return sample_x, sample_y

    def __len__(self):
        return len(self.x)
