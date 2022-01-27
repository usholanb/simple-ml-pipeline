from typing import AnyStr, Dict
from abc import abstractmethod
import pandas as pd
import torch
from modules.containers.di_containers import TrainerContainer
from modules.helpers.csv_saver import CSVSaver
from modules.wrappers.base_wrappers.base_wrapper import BaseWrapper
from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper
from utils.common import unpickle_obj, create_folder
from utils.constants import CLASSIFIERS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR
from copy import deepcopy

from utils.registry import registry


class BasePredictor:
    """ uses all wrappers pointed in prediction config to
        make and save metrics """

    @property
    def pred_dir(self) -> AnyStr:
        dataset_name = self.configs.get('dataset').get('input_path').split('/')[-1]
        return create_folder(f'{PREDICTIONS_DIR}/{dataset_name}')

    @abstractmethod
    def get_preds_ys(self) -> Dict:
        """ """

    @abstractmethod
    def get_metrics(self, split: Dict, split_name: AnyStr) -> Dict:
        """ Saves metrics for the split  """

    @abstractmethod
    def save_results(self, preds_ys: Dict) -> None:
        """ saves splits with predictions and metrics """

    @abstractmethod
    def save_graphs(self, output_dataset: Dict) -> None:
        """ override if need graphs """

