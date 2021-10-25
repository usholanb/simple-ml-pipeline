from modules.containers.train_container import TrainContainer
from utils.flags import train_flags
from dependency_injector.wiring import Provide, inject
from utils.common import build_config, setup_imports
from utils.registry import registry


@inject
def train():
    """ Prepares Dataset """
    setup_imports()
    config = Provide[TrainContainer.config]
    dataset = registry.get_dataset_class(config.get('dataset').get('name'))(config)
    model = registry.get_model_class(config.get('model').get('name'))(config)
    trainer = registry.get_trainer_class(config.get('model').get('name'))(config)
    trainer.train(dataset, model)


if __name__ == '__main__':
    setup_logging()
    parser = train_flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args)
    train_container = TrainContainer()
    train_container.config.from_dict(config)
    train()
