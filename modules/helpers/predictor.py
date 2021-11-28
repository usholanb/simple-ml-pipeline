import pandas as pd

from modules.helpers.csv_saver import CSVSaver
from utils.common import unpickle_obj
from utils.constants import CLASSIFIERS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR
from copy import deepcopy

from utils.registry import registry


class Predictor:
    """ uses all wrappers pointed in prediction config to
        make and save several prediction files """

    def __init__(self, configs, dataset):
        self.configs = configs
        self.dataset = dataset

    def predict(self) -> pd.DataFrame:
        output_dataset = deepcopy(self.dataset)
        for tag, model_name in self.configs.get('models').items():
            model_name_tag = f'{model_name}_{tag}'
            model_path = f'{CLASSIFIERS_DIR}/{model_name_tag}.pkl'
            wrapper = unpickle_obj(model_path)
            probs = wrapper.predict_proba(self.dataset)
            if len(wrapper.label_types) > 1:
                for label, label_index in wrapper.label_types.items():
                    output_dataset[f'{model_name_tag}_{label}'] = probs[:, label_index]
            else:
                output_dataset[f'{model_name_tag}'] = probs
        return output_dataset

    def save_metrics(self, split, split_name, dataset_name):
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
        CSVSaver.save_file(f'{PREDICTIONS_DIR}/{dataset_name}_{split_name}_metrics',
                           df, index=True, compression=None)

    def save_results(self, output_dataset) -> None:
        for split_name in output_dataset['split'].unique():
            split = output_dataset.loc[output_dataset['split'] == split_name]
            dataset_path = self.configs.get('dataset').get('input_path')
            dataset_name = dataset_path.split('/')[1]
            CSVSaver.save_file(f'{PREDICTIONS_DIR}/{dataset_name}_{split_name}', split)
            self.save_metrics(split, split_name, dataset_name)

