from typing import Dict
from modules.models.base_model import BaseModel


class DefaultModel(BaseModel):

    def __init__(self, configs: Dict):
        self.config = configs
        self.clf = self.get_classifier(configs.get('optim').get('search_space'))

    @property
    def name(self):
        return self.config.get('model').get('name')
