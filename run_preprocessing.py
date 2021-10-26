from modules.containers.di_containers import ConfigContainer
from modules.containers.di_containers import SaverContainer
from utils.flags import preprocessing_flags
from dependency_injector.wiring import Provide, inject
from utils.common import build_config, setup_imports
from utils.registry import registry
from typing import Dict


@inject
def preprocessing(config: Dict = ConfigContainer.config) -> None:
    """ Prepares Dataset """
    setup_imports()
    dataset = registry.get_dataset_class(config.get('dataset').get('name'))()
    dataset.collect()
    dataset.save(saver=SaverContainer.csv_saver())
    print(f'{dataset.name} is ready')


if __name__ == '__main__':
    # setup_logging()
    parser = preprocessing_flags.parser
    args = parser.parse_args()
    yml_configs = build_config(args)
    ConfigContainer.config.from_dict(yml_configs, required=True)
    preprocessing()
