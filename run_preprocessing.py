"""
This script utilizes one of the readers and processes the data converting
it to gz

input reader is in data/ folder or/and from remote server
output reader is is processed_data/ folder
"""
from modules.helpers.csv_saver import CSVSaver
from utils.common import build_config, setup_imports, setup_directories
from utils.registry import registry
from typing import Dict, AnyStr


def preprocessing(configs: Dict) -> None:
    """ Prepares Reader """
    setup_imports()
    reader = registry.get_reader_class(
        configs.get('reader').get('name'))(configs)
    reader.collect()
    CSVSaver().save(reader.data, reader.reader_configs)
    print(f'{reader.name} is ready, reader: {reader.data.shape}')


def run_preprocessing(k_fold_tag: AnyStr = ''):
    print('started run_preprocessing')
    setup_directories()
    from utils.flags import all_flags
    parser = all_flags['preprocessing'].parser
    args = parser.parse_args()
    configs = build_config(args)
    configs['reader']['k_fold_tag'] = k_fold_tag
    preprocessing(configs)


if __name__ == '__main__':
    run_preprocessing()
