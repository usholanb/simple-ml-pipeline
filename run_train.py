from utils.constants import TRAIN_RESULTS_DIR, CLASSIFIERS_DIR
from utils.flags import train_flags
from utils.common import build_config, setup_imports, setup_directories, add_grid_search_parameters
from utils.registry import registry
import pandas as pd
from ray import tune
from functools import partial
from typing import Dict


def train(configs, dataset):
    """ Prepares Dataset """
    setup_imports()
    trainer = registry.get_trainer_class(
        configs.get('trainer').get('name')
    )(configs, dataset)
    trainer.train()


def grid_search(configs: Dict):
    dataset = pd.read_csv(configs.get('dataset').get('input_path'))
    grid = add_grid_search_parameters(configs)
    sync_config = tune.SyncConfig(sync_to_driver=False)

    if grid:
        analysis = tune.run(
            tune.with_parameters(train, dataset=dataset),
            config=configs,
            local_dir=TRAIN_RESULTS_DIR,
            resources_per_trial=configs.get('trainer').get('resources_per_trial'),
            sync_config=sync_config,

        )
        best_configs = analysis.get_best_config(metric="test_loss", mode="min")
        print("Best configs: ", best_configs.get('optim'))
    else:
        train(configs, dataset)


if __name__ == '__main__':
    # setup_logging()
    setup_directories()
    parser = train_flags.parser
    args = parser.parse_args()
    config = build_config(args)
    grid_search(config)
