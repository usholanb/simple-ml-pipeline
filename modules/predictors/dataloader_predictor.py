from typing import AnyStr, Dict
import pandas as pd
import torch
from modules.containers.di_containers import TrainerContainer
from modules.helpers.matplotlibgraph import MatPlotLibGraph
from modules.predictors.base_predictors.base_predictor import BasePredictor
from modules.trainers.default_trainer import metrics_fom_torch
from utils.common import unpickle_obj, transform, mean_dict_values, get_data_loaders
from utils.constants import CLASSIFIERS_DIR
from utils.registry import registry
import numpy as np
import csv
import utils.small_functions as sf


@registry.register_predictor('dataloader_predictor')
class DataloaderPredictor(BasePredictor):
    """ uses all wrappers pointed in prediction config to
        make and save several prediction files

        Compares multi prediction models
     """

    def __init__(self, configs: Dict):
        super().__init__(configs)
