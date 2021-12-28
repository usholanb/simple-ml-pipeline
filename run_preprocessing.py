"""
This script utilizes one of the datasets and processes the data converting
it to gz

input dataset is in data/ folder or/and from remote server
output dataset is is processed_data/ folder
"""
from modules.helpers.csv_saver import CSVSaver
from utils.common import build_config, setup_imports, setup_directories
from utils.registry import registry
from typing import Dict, AnyStr


def preprocessing(configs: Dict) -> None:
    """ Prepares Dataset """
    setup_imports()
    dataset = registry.get_dataset_class(
        configs.get('dataset').get('name'))(configs)
    dataset.collect()
    CSVSaver().save(dataset.data, configs)
    print(f'{dataset.name} is ready, dataset: {dataset.data.shape}')


def run_preprocessing(k_fold_tag: AnyStr = ''):
    print('started run_preprocessing')
    setup_directories()
    from utils.flags import all_flags
    parser = all_flags['preprocessing'].parser
    args = parser.parse_args()
    configs = build_config(args)
    configs['dataset']['k_fold_tag'] = k_fold_tag
    preprocessing(configs)


if __name__ == '__main__':
    run_preprocessing()
