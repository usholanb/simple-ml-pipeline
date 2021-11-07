from typing import Dict

from modules.helpers.csv_saver import CSVSaver
from modules.helpers.predictor import Predictor
from utils.flags import train_flags, prediction_flags
from dependency_injector.wiring import Provide, inject
from utils.common import build_config, setup_imports, setup_directories
import pandas as pd


def prediction(configs: Dict):
    """ Prepares Dataset """
    setup_imports()
    dataset = CSVSaver().load(configs)
    predictor = Predictor(configs, dataset)
    output_dataset = predictor.predict()
    predictor.save_probs(output_dataset)


if __name__ == '__main__':
    setup_directories()
    parser = prediction_flags.parser
    args = parser.parse_args()
    configs = build_config(args)
    prediction(configs)
