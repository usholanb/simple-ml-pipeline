from modules.helpers.predictor import Predictor
from utils.flags import train_flags, prediction_flags
from dependency_injector.wiring import Provide, inject
from utils.common import build_config, setup_imports, setup_directories
from utils.registry import registry
import pandas as pd


def prediction(configs):
    """ Prepares Dataset """
    setup_imports()
    dataset = pd.read_csv(configs.get('dataset').get('input_path'))
    predictor = Predictor(configs, dataset)
    output_dataset = predictor.predict()
    predictor.save_probs(output_dataset)


if __name__ == '__main__':
    # setup_logging()
    setup_directories()
    parser = prediction_flags.parser
    args = parser.parse_args()
    configs = build_config(args)
    prediction(configs)
