import torch
from modules.datasets.base_datasets.default_dataset import DefaultDataset
from modules.helpers.csv_saver import CSVSaver
from utils.common import prepare_train
from utils.registry import registry


@registry.register_dataset('pandas_dataset')
class PandasDataset(DefaultDataset):
    def __init__(self, configs, split_name):
        super(PandasDataset, self).__init__(configs, split_name)
        data = prepare_train(configs, CSVSaver.load(configs))
        self.x = data[f'{split_name}_x'].values
        self.y = data[f'{split_name}_y'].values
        self.shape_dataset()

    def shape_dataset(self):
        self.x = torch.FloatTensor(self.x)
        self.y = torch.Tensor(self.y).reshape((len(self.x), -1))
        self.y = self.y.reshape(self.batch_size, -1) \
            if self.y.shape[1] > 1 else self.y.flatten()
        if self.classification:
            self.y = self.y.long()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
