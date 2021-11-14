import pandas as pd

from modules.containers.di_containers import TrainerContainer
from modules.helpers.csv_saver import CSVSaver
from utils.common import unpickle_obj, get_data_loaders, transform
from utils.constants import CLASSIFIERS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR
from copy import deepcopy
import torch
from utils.registry import registry


class Predictor:
    """ uses all wrappers pointed in prediction config to
        make and save several prediction files """

    def __init__(self, configs):
        self.device = TrainerContainer.device
        self.configs = configs
        dls = get_data_loaders(configs, specific=None)
        self.train_loader, self.valid_loader, self.test_loader = dls

    def predict(self):
        for tag, model_name in self.configs.get('models').items():
            model_name_tag = f'{model_name}_{tag}'
            model_path = f'{CLASSIFIERS_DIR}/{model_name_tag}.pkl'
            model = unpickle_obj(model_path)
            model.to(self.device)
            model.eval()
            model.before_epoch_eval()
            with torch.no_grad():
                for batch_i, batch in enumerate(self.test_loader):
                    data = [x.to(self.device) for x in batch]
                    transformed_data = transform(data, self.configs)
                    forward_data = model.before_iteration_eval(transformed_data)
                    outputs = model.forward(forward_data)
                    model.end_iteration_compute_predictions(data, forward_data, outputs)
                predictions_results = model.after_epoch_predictions('test', self.test_loader)
            print(f'{model_name_tag}: {predictions_results}')
            return predictions_results

    def save_probs(self, output_dataset) -> None:
        for split_name in output_dataset['split'].unique():
            split = output_dataset.loc[output_dataset['split'] == split_name]
            dataset_path = self.configs.get('dataset').get('input_path')
            dataset_name = dataset_path.split('_output.csv')[0].split('/')[1]
            CSVSaver.save_file(f'{PREDICTIONS_DIR}/{dataset_name}_{split_name}', split)

