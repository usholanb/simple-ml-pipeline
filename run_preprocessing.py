"""
This script utilizes one of the datasets and processes the data converting
it to gz

input dataset is in data/ folder or/and from remote server
output dataset is is processed_data/ folder
"""
from modules.helpers.csv_saver import CSVSaver
from modules.helpers.namer import Namer
from utils.flags import preprocessing_flags
from utils.common import build_config, setup_imports, setup_directories
from utils.registry import registry
from typing import Dict


def preprocessing(configs: Dict) -> None:
    """ Prepares Dataset """
    setup_imports()
    dataset = registry.get_dataset_class(configs.get('dataset').get('name'))(configs)
    dataset.collect()
    CSVSaver().save(dataset.data, configs)
    print(f'{Namer.dataset_name(configs)} is ready')


if __name__ == '__main__':
    setup_directories()
    parser = preprocessing_flags.parser
    args = parser.parse_args()
    configs = build_config(args)

    preprocessing(configs)
