from typing import AnyStr, Dict
from abc import abstractmethod
import pandas as pd
import torch
from modules.containers.di_containers import TrainerContainer
from modules.helpers.csv_saver import CSVSaver
from modules.wrappers.base_wrappers.base_wrapper import BaseWrapper
from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper
from utils.common import unpickle_obj, create_folder, get_data_loaders, mean_dict_values
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

    def print_important_features(self, wrapper: DefaultWrapper) -> None:
        if self.configs.get('print_important_features', False):
            self.feature_importance = {
                k: v for k, v in
                zip(wrapper.features_list, wrapper.clf.feature_importances_)
            }
            print(sorted(self.feature_importance.items(), key=lambda x: -x[1]))

    @abstractmethod
    def get_preds_ys(self) -> Dict:
        """ """

    @abstractmethod
    def save_metrics(self, split: Dict, split_name: AnyStr) -> Dict:
        """ Saves metrics for the split  """

    @abstractmethod
    def save_results(self, preds_ys: Dict) -> None:
        """ saves splits with predictions and metrics """

    @abstractmethod
    def save_graphs(self, output_dataset: Dict) -> None:
        """ override if need graphs """

    @abstractmethod
    def predict_dataset(self, wrapper: BaseWrapper) -> Dict:
        """ Return metrics for each split for given wrapper """
