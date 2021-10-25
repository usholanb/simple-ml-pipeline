from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    @abstractmethod
    def train(self, dataset: BaseDataset, model: BaseModel):
        """ trains the model with dataset """
