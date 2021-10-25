from modules.containers.preprocessing_container import PreprocessingContainer
from utils.flags import preprocessing_flags
from dependency_injector.wiring import Provide, inject
from utils.common import build_config, setup_imports
from utils.registry import registry


@inject
def preprocessing():
    """ Prepares Dataset """
    setup_imports()
    config = Provide[PreprocessingContainer.config]
    dataset = registry.get_dataset_class(config.get('dataset').get('name'))(config)
    dataset.collect()
    dataset.save()
    print(f'{dataset.name} is ready')


if __name__ == '__main__':
    # setup_logging()
    parser = preprocessing_flags.parser
    args, override_args = parser.parse_known_args()
    config = build_config(args)
    preprocessing_container = PreprocessingContainer()
    preprocessing_container.config.from_dict(config)
    preprocessing()
