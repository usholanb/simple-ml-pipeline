"""
this scripts makes one (useful for debugging) or multiple
training of the same model with different hyper-parameters

The best model can be saved

All details of the training are specified in the config file

"""
from copy import deepcopy

from modules.helpers.csv_saver import CSVSaver
from utils.constants import TRAIN_RESULTS_DIR
from utils.flags import train_flags
from utils.common import build_config, setup_imports, setup_directories, add_grid_search_parameters
from utils.registry import registry
import pandas as pd
from ray import tune
from typing import Dict


def train_one(configs: Dict, dataset: pd.DataFrame, save: bool = False) -> None:
    """ Prepares Dataset """
    setup_imports()
    trainer = registry.get_trainer_class(
        configs.get('trainer').get('name')
    )(configs, dataset)
    trainer.train()
    if save:
        trainer.save()


def train(configs: Dict) -> None:
    dataset = CSVSaver().load(configs)
    grid = add_grid_search_parameters(configs)
    sync_config = tune.SyncConfig(sync_to_driver=False)
    config_copy = deepcopy(configs)
    m = configs.get("model")
    if grid:
        analysis = tune.run(
            tune.with_parameters(train_one, dataset=dataset),
            config=configs,
            local_dir=TRAIN_RESULTS_DIR,
            sync_config=sync_config,
            name=f'{m.get("name")}_{m.get("tag")}',
            **config_copy.get('trainer').get('tune', {}),

        )
        best_configs = analysis.get_best_config(
            metric=configs.get('trainer').get('grid_metric').get('name'),
            mode=configs.get('trainer').get('grid_metric').get('mode')
        )

        print("Best configs: ", {**best_configs.get('optim'),
                                 **best_configs.get('special_inputs')})
        configs = best_configs

    train_one(configs, dataset, save=configs.get('trainer').get('save'))


if __name__ == '__main__':
    setup_directories()
    parser = train_flags.parser
    args = parser.parse_args()
    config = build_config(args)
    train(config)
