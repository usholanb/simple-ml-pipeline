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
        make and save several prediction files """

    def __init__(self, configs: Dict):
        self.configs = configs
        self.device = TrainerContainer.device
        self.train_loader, self.valid_loader, self.test_loader = \
            [None] * 3
        self.feature_importance = None
        self.split_names = self.configs.get('splits', [])

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

    def get_preds_ys(self) -> Dict:
        model_results = {}
        for tag, model_name in self.configs.get('models').items():
            k_fold_tag = self.configs.get('dataset').get('k_fold_tag', '')
            model_name_tag = f'{model_name}_{tag}{k_fold_tag}'
            model_path = f'{CLASSIFIERS_DIR}/{model_name_tag}.pkl'
            wrapper = unpickle_obj(model_path)
            model_results[model_name_tag] = \
                wrapper.predict_dataset(self.configs, self.split_names)
        return model_results

    def save_metrics(self, split: Dict, split_name: AnyStr) -> Dict:
        """ Saves metrics for the split  """
        y_true = split[f'{split_name}_ys']
        metrics_values = {}
        for metric_name in self.configs.get('metrics', []):
            metric = registry.get_metric_class(metric_name)()
            y_outputs = split[f'{split_name}_preds']
            values = metric.compute_metric(y_true, y_outputs)
            metrics_values[metric_name] = values
        return metrics_values

    def save_predictions(self, split: pd.DataFrame, split_name: AnyStr, dataset_name: AnyStr) -> None:
        CSVSaver.save_file(f'{self.pred_dir}/{dataset_name}_{split_name}', split)

    def save_results(self, preds_ys: Dict) -> None:
        """ saves splits with predictions and metrics """
        metrics = {}
        dataset_path = self.configs.get('dataset').get('input_path')
        dataset_name = dataset_path.split('/')[1]
        for model_name, splits in preds_ys.items():
            metrics[model_name] = {}
            for split_name, split in splits.items():
                metrics[model_name][split_name] = self.save_metrics(split, split_name)
        df = pd.concat({k: pd.DataFrame(v) for k, v in metrics.items()})
        CSVSaver.save_file(f'{self.pred_dir}/{dataset_name}_metrics',
                           df, gzip=False, index=True)

    def save_graphs(self, output_dataset: Dict) -> None:
        """ override if need graphs """

    @abstractmethod
    def predict_dataset(self, wrapper: BaseWrapper) -> Dict:
        """ Return metrics for each split for given wrapper """
