from typing import Dict, List
from modules.wrappers.base_wrappers.base_wrapper import BaseWrapper


class DefaultWrapper(BaseWrapper):

    def __init__(self, configs: Dict, label_types: List):
        self.config = configs
        self.label_types = label_types
        self.clf = self.get_classifier(configs.get('special_inputs'))

    @property
    def name(self):
        return self.config.get('model').get('name')
