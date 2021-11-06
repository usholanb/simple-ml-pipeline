from copy import deepcopy

from utils.constants import TRAIN_RESULTS_DIR, CLASSIFIERS_DIR
from utils.flags import train_flags
from utils.common import build_config, setup_imports, setup_directories, add_grid_search_parameters
from utils.registry import registry
import pandas as pd
from ray import tune
from functools import partial
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
    dataset = pd.read_csv(configs.get('dataset').get('input_path'))
    grid = add_grid_search_parameters(configs)
    sync_config = tune.SyncConfig(sync_to_driver=False)
    configs_copy = deepcopy(configs)
    if grid:
        analysis = tune.run(
            tune.with_parameters(train_one, dataset=dataset),
            config=configs_copy,
            local_dir=TRAIN_RESULTS_DIR,
            sync_config=sync_config,
            **configs_copy.get('trainer').get('tune'),
            name='my_exp',

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
