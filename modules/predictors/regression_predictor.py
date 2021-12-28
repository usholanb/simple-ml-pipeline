from typing import AnyStr, Dict
import pandas as pd
import torch
from modules.containers.di_containers import TrainerContainer
from modules.helpers.matplotlibgraph import MatPlotLibGraph
from modules.predictors.base_predictors.base_predictor import BasePredictor
from utils.common import unpickle_obj, transform
from utils.constants import CLASSIFIERS_DIR
from utils.registry import registry
import numpy as np
import csv
import utils.small_functions as sf


@registry.register_predictor('regression_predictor')
class RegressionPredictor(BasePredictor):
    """ uses all wrappers pointed in prediction config to
        make and save several prediction files

        Compares multi prediction models
     """

    def __init__(self, configs: Dict):
        super().__init__(configs)

    def make_predict(self):
        model_results = {}
        for tag, model_name in self.configs.get('models').items():
            model_name_tag = f'{model_name}_{tag}'
            model_path = f'{CLASSIFIERS_DIR}/{model_name_tag}.pkl'
            model = unpickle_obj(model_path)
            model.to(self.device)
            model.eval()
            with torch.no_grad():
                for batch_i, batch in enumerate(self.test_loader):
                    x, y = model.get_x_y(batch)
                    all_data = {
                        'epoch': 0,
                        'batch_i': batch_i,
                        'x': x,
                        'split': 'test',
                        'batch_size': self.test_loader.dataset.batch_size
                    }
                    predictions_results = model.forward(all_data)
                model_results[model_name_tag] = predictions_results
            print(f'{model_name_tag}: {predictions_results}')
        return model_results

