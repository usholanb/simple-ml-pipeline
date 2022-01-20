from typing import AnyStr, Dict
from abc import abstractmethod
import pandas as pd
import torch
from modules.containers.di_containers import TrainerContainer
from modules.helpers.csv_saver import CSVSaver
from modules.predictors.base_predictors.base_predictor import BasePredictor
from modules.wrappers.base_wrappers.base_wrapper import BaseWrapper
from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper
from utils.common import unpickle_obj, create_folder
from utils.constants import CLASSIFIERS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR
from copy import deepcopy

from utils.registry import registry


@registry.register_predictor('simple_predictor')
class SimplePredictor(BasePredictor):
    """ uses all wrappers pointed in prediction config to
        make and save metrics """

    def __init__(self, configs: Dict):
        self.configs = configs
        self.device = TrainerContainer.device
        self.train_loader, self.valid_loader, self.test_loader = \
            [None] * 3
        self.split_names = self.configs.get('splits', [])

    @property
    def pred_dir(self) -> AnyStr:
        dataset_path = self.configs.get('dataset').get('input_path')
        if dataset_path is not None:
            dataset_name = dataset_path.split('/')[1]
        else:
            dataset_name = self.configs.get('dataset').get('name')
        return create_folder(f'{PREDICTIONS_DIR}/{dataset_name}')

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

    def get_metrics(self, split: Dict, split_name: AnyStr) -> Dict:
        """ Saves metrics for the split  """
        y_true = split[f'{split_name}_ys']
        metrics_values = {}
        for metric_name in self.configs.get('metrics', []):
            metric = registry.get_metric_class(metric_name)()
            y_outputs = split[f'{split_name}_preds']
            values = metric.compute_metric(y_true, y_outputs)
            metrics_values[metric_name] = values
        return metrics_values

    def save_metrics(self, preds_ys: Dict) -> None:
        """ """
        metrics = {}
        for model_name, splits in preds_ys.items():
            metrics[model_name] = {}
            for split_name, split in splits.items():
                metrics[model_name][split_name] = self.get_metrics(split, split_name)
        df = pd.concat({k: pd.DataFrame(v) for k, v in metrics.items()})
        CSVSaver.save_file(f'{self.pred_dir}/metrics',
                           df, gzip=False, index=True)

    def yield_model_path(self):
        for tag, model_name in self.configs.get('models').items():
            k_fold_tag = self.configs.get('dataset').get('k_fold_tag', '')
            model_name_tag = f'{model_name}_{tag}{k_fold_tag}'
            model_path = f'{CLASSIFIERS_DIR}/{model_name_tag}.pkl'
            yield model_path, model_name_tag

    def save_predictions(self, preds_ys):
        if self.configs.get('dataset').get('input_path', None):
            data = CSVSaver().load(self.configs)
            for split_name in self.configs.get('splits', []):
                split = data[data['split'] == split_name]
                for model_path, model_name_tag in self.yield_model_path():
                    preds = preds_ys[model_name_tag][split_name][f'{split_name}_preds']
                    split.insert(len(split.columns), model_name_tag, preds, False)
                CSVSaver.save_file(f'{self.pred_dir}/predictions_{split_name}', split, gzip=True, index=False)

    def save_graphs(self, output_dataset: Dict) -> None:
        """ override if need graphs """

