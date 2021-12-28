from torch.utils.data import Dataset


class DefaultDataset(Dataset):

    def __init__(self, configs, split_name):
        self.configs = configs
        self.batch_size = configs.get('dataset').get('data_loaders', {})\
            .get('split_name', {}).get('batch_size', 32)
        self.split_name = split_name

    @property
    def classification(self):
        return self.configs.get('dataset').get('label_type') == 'classification'

    @property
    def regression(self):
        return self.configs.get('dataset').get('label_type') == 'regression'

