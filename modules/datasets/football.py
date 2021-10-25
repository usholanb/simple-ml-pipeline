from modules.datasets.base_dataset import BaseDataset
from modules.interfaces.saver import Saver
from utils.registry import registry


@registry.register_model('football')
class Football(BaseDataset):
    """ Footlball specific dataset """
