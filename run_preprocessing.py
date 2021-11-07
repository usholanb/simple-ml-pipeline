"""
This script utilizes one of the datasets and processes the data converting
it to gz
"""
from modules.helpers.csv_saver import CSVSaver
from utils.flags import preprocessing_flags
from dependency_injector.wiring import Provide, inject
from utils.common import build_config, setup_imports, setup_directories
from utils.registry import registry
from typing import Dict


@inject
def preprocessing(configs: Dict) -> None:
    """ Prepares Dataset """
    setup_imports()
    dataset = registry.get_dataset_class(configs.get('dataset').get('name'))(configs)
    dataset.collect()
    CSVSaver().save(dataset, configs)
    print(f'{dataset.name} is ready')


if __name__ == '__main__':
    # setup_logging()
    setup_directories()
    parser = preprocessing_flags.parser
    args = parser.parse_args()
    configs = build_config(args)

    preprocessing(configs)
