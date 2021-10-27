from modules.datasets.base_dataset import BaseDataset
from modules.models.base_model import BaseModel
from modules.trainers.base_trainer import BaseTrainer
from modules.trainers.default_trainer import DefaultTrainer
from utils.registry import registry


@registry.register_trainer('nn_trainer')
class NNTrainer(DefaultTrainer):

    def train(self, config) -> None:
        """ trains nn model with dataset """
