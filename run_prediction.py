"""
This script uses the models specified in the config and gets the predictions
on the specified in the config dataset

The predictions and probabilities are saved in the prediction files in
predictions/ folder
"""
from typing import Dict, AnyStr

import pandas as pd

from modules.helpers.csv_saver import CSVSaver
from modules.predictors.base_predictors.predictor import Predictor
from utils.common import build_config, setup_imports, setup_directories
from utils.registry import registry


def get_predictor(k_fold_tag: AnyStr) -> Predictor:
    setup_directories()
    from utils.flags import all_flags
    parser = all_flags['prediction'].parser
    args = parser.parse_args()
    configs = build_config(args)
    configs['dataset']['k_fold_tag'] = k_fold_tag
    setup_imports()
    dataset = CSVSaver().load(configs)
    predictor_name = configs.get('predictor', 'predictor')
    return registry.get_predictor_class(predictor_name)(configs, dataset)


def save_files(predictor: Predictor, output_dataset: pd.DataFrame) -> None:
    """ Prepares Dataset """
    # predictor.save_results(output_dataset)
    predictor.save_graphs(output_dataset)


def run_prediction(k_fold_tag: AnyStr = '') -> None:
    print('started run_prediction')
    predictor = get_predictor(k_fold_tag)
    output_dataset = predictor.predict()
    save_files(predictor, output_dataset)


if __name__ == '__main__':
    run_prediction()
