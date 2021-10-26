from modules.datasets.base_dataset import BaseDataset
from utils.registry import registry


@registry.register_dataset('football')
class Football(BaseDataset):
    """ Footlball specific dataset """
