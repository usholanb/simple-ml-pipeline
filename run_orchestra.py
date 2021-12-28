"""
This script uses the models specified in the config and gets the predictions
on the specified in the config dataset

The predictions and probabilities are saved in the prediction files in
predictions/ folder
"""
from typing import Dict
import pandas as pd
from modules.helpers.csv_saver import CSVSaver
from modules.predictors.base_predictors.base_predictor import BasePredictor
from run_preprocessing import run_preprocessing
from run_train import run_train
from run_prediction import run_prediction, save_files, get_predictor
from utils.common import build_config, setup_imports, setup_directories
from utils.flags import all_flags, CustomFlag
from utils.registry import registry


def orchestra(configs: Dict):
    setup_imports()
    preprocessing_config = configs.get('preprocessing')
    flag = CustomFlag('preprocessing').add_config_args(preprocessing_config)
    all_flags['preprocessing'] = flag
    output_datasets = []
    for k_fold_i in range(configs.get('k_fold')):
        k_fold_tag = f'k_fold{k_fold_i}'
        # run_preprocessing(k_fold_tag=k_fold_tag)
        # for train_config in configs.get('train'):
        #     flag = CustomFlag('training').add_config_args(train_config)
        #     all_flags['train'] = flag
        #     run_train(k_fold_tag=k_fold_tag)

        prediction_config = configs.get('prediction')
        flag = CustomFlag('prediction').add_config_args(prediction_config)
        all_flags['prediction'] = flag
        output_dataset = get_predictor(k_fold_tag).make_predict()
        output_datasets.append(output_dataset)
    all_folds = pd.concat(output_datasets)
    predictor_name = configs.get('predictor')
    predictor = registry.get_predictor_class(predictor_name)(configs, all_folds)
    save_files(predictor, all_folds)


def run_orchestra():
    setup_directories()
    parser = all_flags['orchestra'].parser
    args = parser.parse_args()
    configs = build_config(args)
    orchestra(configs)


if __name__ == '__main__':
    run_orchestra()
