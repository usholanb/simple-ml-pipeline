from typing import Dict, List
from modules.models.base_models.base_model import BaseModel


class DefaultModel(BaseModel):

    def __init__(self, configs: Dict, label_types: List):
        self.config = configs
        self.label_types = label_types
        self.clf = self.get_classifier(configs.get('optim').get('search_space'))

    @property
    def name(self):
        return self.config.get('model').get('name')
