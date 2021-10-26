from abc import ABC, abstractmethod


class Saver(ABC):

    @abstractmethod
    def save(self, data, config):
        """ save to some format """

