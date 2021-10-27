from abc import ABC, abstractmethod

from modules.datasets.base_dataset import BaseDataset
from modules.models.base_model import BaseModel


class BaseTrainer(ABC):

    @abstractmethod
    def train(self) -> None:
        """ trains the model with dataset """

    @abstractmethod
    def ce_loss(self, targets, probs):
        """ returns Cross Entropy loss """
