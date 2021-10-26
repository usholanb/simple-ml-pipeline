from abc import ABC, abstractmethod

from modules.datasets.base_dataset import BaseDataset
from modules.models.base_model import BaseModel


class BaseTrainer(ABC):

    @abstractmethod
    def train(self, dataset, model: BaseModel):
        """ trains the model with dataset """
