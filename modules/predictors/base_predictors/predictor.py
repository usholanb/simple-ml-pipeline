from typing import AnyStr, Dict
from abc import abstractmethod
import pandas as pd

from modules.containers.di_containers import TrainerContainer
from modules.helpers.csv_saver import CSVSaver
from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper
from utils.common import unpickle_obj, create_folder, get_data_loaders
from utils.constants import CLASSIFIERS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR
from copy import deepcopy

from utils.registry import registry


class Predictor:
    """ uses all wrappers pointed in prediction config to
        make and save several prediction files """

    def __init__(self, configs: Dict):
        self.configs = configs
        self.device = TrainerContainer.device
        self.get_data()

    def get_data(self):
        if self.configs.get('dataset').get('dataloaders', True):
            self.train_loader, self.valid_loader, self.test_loader = \
                get_data_loaders(self.configs)
        else:
            setattr(self, 'dataset', CSVSaver().load(self.configs))

    @property
    def pred_dir(self):
        dataset_name = self.configs.get('dataset').get('input_path').split('/')[-1]
        return create_folder(f'{PREDICTIONS_DIR}/{dataset_name}')

    def print_important_features(self, wrapper: DefaultWrapper) -> None:
        if self.configs.get('print_important_features', False):
            self.feature_importance = {k: v for k, v in zip(wrapper.features_list, wrapper.clf.feature_importances_)}
            print(sorted(self.feature_importance.items(), key=lambda x: -x[1]))

    @abstractmethod
    def predict(self):
        """ appends probs for each class for each model """

    def save_metrics(self, split: pd.DataFrame, split_name: AnyStr, dataset_name: AnyStr) -> None:
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

    def save_results(self, output_dataset: pd.DataFrame) -> None:
        for split_name in output_dataset['split'].unique():
            split = output_dataset.loc[output_dataset['split'] == split_name]
            dataset_path = self.configs.get('dataset').get('input_path')
            dataset_name = dataset_path.split('/')[1]
            CSVSaver.save_file(f'{self.pred_dir}/{dataset_name}_{split_name}', split)
            self.save_metrics(split, split_name, dataset_name)

    def save_graphs(self, output_dataset: pd.DataFrame):
        """ saves various project specific graphs """

