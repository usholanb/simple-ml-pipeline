import sys
from typing import Dict, List, AnyStr

import numpy as np
import pandas as pd

from modules.containers.di_containers import TrainerContainer
from modules.helpers.namer import Namer
from modules.helpers.z_score import ZScore
from modules.wrappers.base_wrappers.base_wrapper import BaseWrapper
import importlib

from utils.common import is_outside_library, to_snake_case
from utils.constants import CLASSIFIERS_DIR


class DefaultWrapper(BaseWrapper):

    def __init__(self, configs: Dict):
        self.device = TrainerContainer.device
        self.configs = configs
        self.clf = self.get_classifier(configs)
        self._features_list = self.configs.get('features_list', [])
        self.__dict__.update(self.configs.get('special_inputs', {}))

    @property
    def features_list(self):
        return self._features_list

    @property
    def model_path(self) -> AnyStr:
        return f'{CLASSIFIERS_DIR}/{self.name}.pkl'

    @property
    def name(self) -> AnyStr:
        kfold = self.configs.get('dataset', {}).get('k_fold_tag', '')
        return f"{Namer().model_name(self.configs.get('model', {}))}{kfold}"

    def filter_features(self, examples: pd.DataFrame) -> pd.DataFrame:
        """ picks certain features """
        if self._features_list:
            examples = examples[self._features_list]
        else:
            examples = examples.iloc[:, len(self.configs.get('static_columns')):]
        if len(examples.shape) == 1:
            examples = examples.reshape((1, -1))
        return examples
