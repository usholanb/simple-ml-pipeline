from modules.containers.train_container import TrainContainer
from utils.flags import train_flags
from dependency_injector.wiring import Provide, inject
from utils.common import build_config, setup_imports
from utils.registry import registry
import pandas as pd


@inject
def train():
    """ Prepares Dataset """
    setup_imports()
    config = TrainContainer.config
    dataset = pd.read_csv(config.get('dataset').get('input_path'))
    model = registry.get_model_class(config.get('model').get('name'))(config)
    trainer = registry.get_trainer_class(config.get('model').get('name'))(config)
    trainer.train(dataset, model)


if __name__ == '__main__':
    # setup_logging()
    parser = train_flags.parser
    args = parser.parse_args()
    yml_configs = build_config(args)
    TrainContainer.config.from_dict(yml_configs)
    train()
