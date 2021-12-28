import torch
from modules.datasets.base_datasets.default_dataset import DefaultDataset
from utils.registry import registry


@registry.register_dataset('pandas_dataset')
class PandasDataset(DefaultDataset):
    def __init__(self, x, y, configs, split_name):
        super(PandasDataset, self).__init__(x, y, configs, split_name)
        self.x = torch.FloatTensor(self.x.values)
        self.y = torch.Tensor(self.y.values).reshape((len(self.x), -1))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
