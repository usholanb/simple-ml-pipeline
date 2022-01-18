from typing import Dict, AnyStr, Tuple
import torch
from modules.datasets.base_datasets.default_dataset import DefaultDataset
from modules.helpers.csv_saver import CSVSaver
from utils.common import prepare_train
from utils.constants import PROJECT_DIR
from utils.registry import registry


@registry.register_dataset('pandas_dataset')
class PandasDataset(DefaultDataset):
    """ If no dataset class exists with specified name, then pandas dataset
        is called. The framework assumes there is a csv file with specified
        input_path
    """
    def __init__(self, configs: Dict, split_name: AnyStr):
        super(PandasDataset, self).__init__(configs, split_name)
        data = prepare_train(configs, CSVSaver.load(configs, folder=PROJECT_DIR))
        self.x = data[f'{split_name}_x'].values
        self.y = data[f'{split_name}_y'].values
        self.shape_dataset()

    def shape_dataset(self) -> None:
        """  """
        self.x = torch.FloatTensor(self.x)
        self.y = torch.Tensor(self.y).reshape((len(self.x), -1))
        self.y = self.y.reshape(self.batch_size, -1) \
            if self.y.shape[1] > 1 else self.y.flatten()
        if self.classification:
            self.y = self.y.long()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.x)
