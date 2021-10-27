"""



        for split_name in ['test', 'valid']:
            split = dataset.loc[dataset[split_column] == split_name].iloc[:, 1:-1]
            pred = model.predict(split)
            target = dataset.loc[dataset[split_column] == split_name].iloc[:, 0]



"""


from modules.helpers.predictor import Predictor
from utils.flags import train_flags
from dependency_injector.wiring import Provide, inject
from utils.common import build_config, setup_imports, setup_directories
from utils.registry import registry
import pandas as pd


@inject
def predict(configs):
    """ Prepares Dataset """
    setup_imports()
    dataset = pd.read_csv(configs.get('dataset').get('input_path'))
    predictor = Predictor(configs, dataset)
    predictor.predict()


if __name__ == '__main__':
    # setup_logging()
    setup_directories()
    parser = train_flags.parser
    args = parser.parse_args()
    configs = build_config(args)
    predict(configs)
