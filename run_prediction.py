"""
This script uses the models specified in the config and gets the predictions
on the specified in the config dataset

The predictions and probabilities are saved in the prediction files in
predictions/ folder
"""
from typing import Dict
from modules.helpers.csv_saver import CSVSaver
from modules.predictors.base_predictors.predictor import Predictor
from utils.flags import prediction_flags
from utils.common import build_config, setup_imports, setup_directories
from utils.registry import registry


def prediction(configs: Dict):
    """ Prepares Dataset """
    setup_imports()
    dataset = CSVSaver().load(configs)
    predictor_name = configs.get('predictor', 'predictor')
    predictor = registry.get_predictor_class(predictor_name)(configs, dataset)
    output_dataset = predictor.predict()
    # predictor.save_results(output_dataset)
    predictor.save_graphs(output_dataset)


if __name__ == '__main__':
    setup_directories()
    parser = prediction_flags.parser
    args = parser.parse_args()
    configs = build_config(args)
    prediction(configs)
