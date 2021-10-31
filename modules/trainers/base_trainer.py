from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    @abstractmethod
    def train(self) -> None:
        """ trains the model with dataset """

    @abstractmethod
    def get_loss(self, y_true, y_pred) -> float:
        """ returns some loss func """

    @abstractmethod
    def save(self) -> None:
        """ saves model """
