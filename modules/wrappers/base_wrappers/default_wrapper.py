from typing import Dict, List, AnyStr

import numpy as np
import pandas as pd

from modules.wrappers.base_wrappers.base_wrapper import BaseWrapper
import importlib

from utils.common import is_outside_library, to_snake_case


class DefaultWrapper(BaseWrapper):

    def __init__(self, configs: Dict, label_types: List):

        self.configs = configs
        self.label_types = label_types
        self.clf = self.get_classifier(configs.get('special_inputs', {}))
        self._features_list = self.configs.get('features_list', [])

    @property
    def name(self) -> AnyStr:
        m_configs = self.configs.get("model")
        name = m_configs.get("name")
        if is_outside_library(name):
            name = to_snake_case(name.split('.')[-1])
        return f'{name}_{m_configs.get("tag")}'

    def filter_features(self, examples: pd.DataFrame) -> np.ndarray:
        """ picks certain features and converts to numpy"""
        if self._features_list:
            examples = examples[self._features_list]
        else:
            examples = examples.iloc[:, len(self.configs.get('static_columns')):]
        examples = examples.values.astype(float)
        if len(examples.shape) == 1:
            examples = examples.reshape((1, -1))
        return examples







