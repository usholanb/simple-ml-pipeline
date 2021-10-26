from typing import Dict
from modules.containers.train_container import TrainContainer
from modules.models.base_model import BaseModel


class DefaultModel(BaseModel):

    def __init__(self, config: Dict = TrainContainer.config):
        self.config = config
        self.clf = self.get_classifier()

    @property
    def name(self):
        return self.config.get('model').get('name')
