from torch.utils.data import Dataset


class DefaultDataset(Dataset):

    def __init__(self, x, y, configs, split_name):
        self.configs = configs
        self.batch_size = configs.get('dataset').get('data_loaders', {})\
            .get('split_name', {}).get('batch_size', 32)
        self.x = x
        self.y = y
        self.split_name = split_name


