from modules.trainers.default_trainer import DefaultTrainer
from utils.registry import registry


@registry.register_trainer('nn_trainer')
class NNTrainer(DefaultTrainer):

    def train(self) -> None:
        """ trains nn model with dataset """
