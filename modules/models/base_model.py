from abc import ABC, abstractmethod
from dependency_injector.wiring import Provide
from typing import Dict
from modules.containers.train_container import TrainContainer


class BaseModel(ABC):

    def __init__(self, config: Dict = Provide[TrainContainer.config]):
        self.config = config

    @property
    def name(self):
        return self.config.get('model').get('name')
