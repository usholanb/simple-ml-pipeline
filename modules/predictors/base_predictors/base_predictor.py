from typing import AnyStr, Dict
from abc import abstractmethod
import pandas as pd
import torch
from modules.containers.di_containers import TrainerContainer
from modules.helpers.csv_saver import CSVSaver
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

    @property
    def pred_dir(self):
        dataset_name = self.configs.get('dataset').get('input_path').split('/')[-1]
        return create_folder(f'{PREDICTIONS_DIR}/{dataset_name}')

    def print_important_features(self, wrapper: DefaultWrapper) -> None:
        if self.configs.get('print_important_features', False):
            self.feature_importance = {
                k: v for k, v in
                zip(wrapper.features_list, wrapper.clf.feature_importances_)
            }
            print(sorted(self.feature_importance.items(), key=lambda x: -x[1]))

    def predict_split_model(self, split_x, wrapper, model_name_tag, k_fold_tag=''):
        probs = wrapper.predict_proba(split_x)
        if self.configs.get('classification'):
            if len(wrapper.clf.classes_) > 1:
                for i, label in enumerate(wrapper.clf.classes):
                    pred_name = f'{model_name_tag}{k_fold_tag}_{i}'
                    split_x[pred_name] = probs[:, i]
            else:
                pred_name = f'{model_name_tag}{k_fold_tag}'
                split_x[pred_name] = probs
        else:
            if wrapper.n_outputs > 1:
                for i in range(wrapper.n_outputs):
                    split_x[f'{model_name_tag}{k_fold_tag}_{i}'] = probs[:, i]
            else:
                split_x[f'{model_name_tag}{k_fold_tag}'] = probs[:]
        split_x['k_fold'] = k_fold_tag
        return split_x

    def make_predict(self):
        model_results = {}
        for tag, model_name in self.configs.get('models').items():
            model_name_tag = f'{model_name}_{tag}'
            model_path = f'{CLASSIFIERS_DIR}/{model_name_tag}.pkl'
            wrapper = unpickle_obj(model_path)
            model_results[model_name_tag] = wrapper.predict_dataset()
        return model_results

    def save_metrics(self, split: pd.DataFrame, split_name: AnyStr, dataset_name: AnyStr) -> None:
        """ Saves metrics for the split  """
        y_true_index = self.configs.get('static_columns').get('FINAL_LABEL_INDEX')
        y_true = split.iloc[:, y_true_index].values
        metrics_values = {}
        for metric_name in self.configs.get('metrics'):
            metric = registry.get_metric_class(metric_name)()
            models_values = {}
            for tag, model_name in self.configs.get('models').items():
                model_name_tag = f'{model_name}_{tag}'
                y_outputs = split[model_name_tag].values
                values = metric.compute_metric(y_true, y_outputs)
                models_values[model_name_tag] = values
            metrics_values[metric_name] = models_values
        df = pd.DataFrame(metrics_values)
        CSVSaver.save_file(f'{self.pred_dir}/{dataset_name}_{split_name}_metrics',
                           df, index=True, compression=None)

    def save_predictions(self, split: pd.DataFrame, split_name: AnyStr, dataset_name: AnyStr) -> None:
        CSVSaver.save_file(f'{self.pred_dir}/{dataset_name}_{split_name}', split)

    def save_results(self, output_dataset: pd.DataFrame) -> None:
        """ saves splits with predictions and metrics """
        for split_name in output_dataset['split'].unique():
            split = output_dataset.loc[output_dataset['split'] == split_name]
            dataset_path = self.configs.get('dataset').get('input_path')
            dataset_name = dataset_path.split('/')[1]
            # self.save_predictions(split, split_name, dataset_name)
            self.save_metrics(split, split_name, dataset_name)

    def save_graphs(self, output_dataset: pd.DataFrame):
        """ saves various project specific graphs """

