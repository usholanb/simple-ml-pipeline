from typing import AnyStr, Dict

import pandas as pd

from modules.helpers.csv_saver import CSVSaver
from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper
from utils.common import unpickle_obj, create_folder
from utils.constants import CLASSIFIERS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR
from copy import deepcopy

from utils.registry import registry


@registry.register_predictor('predictor')
class Predictor:
    """ uses all wrappers pointed in prediction config to
        make and save several prediction files """

    def __init__(self, configs: Dict, dataset: pd.DataFrame):
        self.configs = configs
        self.dataset = dataset

    @property
    def pred_dir(self):
        dataset_name = self.configs.get('dataset').get('input_path').split('/')[-1]
        return create_folder(f'{PREDICTIONS_DIR}/{dataset_name}')

    def print_important_features(self, wrapper: DefaultWrapper) -> None:
        if self.configs.get('print_important_features', False):
            self.feature_importance = {k: v for k, v in zip(wrapper.features_list, wrapper.clf.feature_importances_)}
            print(sorted(self.feature_importance.items(), key=lambda x: -x[1]))

    def predict(self) -> pd.DataFrame:
        output_dataset = deepcopy(self.dataset)
        k_fold_tag = self.configs.get('dataset').get('k_fold_tag', '')
        for tag, model_name in self.configs.get('models').items():
            model_name_tag = f'{model_name}_{tag}'
            model_path = f'{CLASSIFIERS_DIR}/{model_name_tag}{k_fold_tag}.pkl'
            wrapper = unpickle_obj(model_path)
            self.print_important_features(wrapper)
            probs = wrapper.predict_proba(self.dataset)
            if len(wrapper.label_types) > 1:
                for label, label_index in wrapper.label_types.items():
                    output_dataset[f'{model_name_tag}_{label}'] = probs[:, label_index]
            else:
                output_dataset[f'{model_name_tag}'] = probs
            output_dataset['k_fold'] = k_fold_tag
        return output_dataset

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

