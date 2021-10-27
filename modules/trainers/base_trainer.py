from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    @abstractmethod
    def train(self) -> None:
        """ trains the model with dataset """

    @abstractmethod
    def ce_loss(self, targets, probs):
        """ returns Cross Entropy loss """
