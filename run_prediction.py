"""
This script uses the models specified in the config and gets the predictions
on the specified in the config dataset

The predictions and probabilities are saved in the prediction files in
predictions/ folder
"""
from typing import Dict
from modules.helpers.csv_saver import CSVSaver
from modules.helpers.predictor import Predictor
from utils.flags import prediction_flags
from utils.common import build_config, setup_imports, setup_directories, get_data_loaders


def prediction(configs: Dict):
    """ Prepares Dataset """
    setup_imports()
    predictor = Predictor(configs)
    output_dataset = predictor.predict()


if __name__ == '__main__':
    setup_directories()
    parser = prediction_flags.parser
    args = parser.parse_args()
    configs = build_config(args)
    prediction(configs)
