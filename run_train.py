"""
this scripts makes one (useful for debugging) or multiple
training of the same model with different hyper-parameters

The best model can be saved

All details of the training are specified in the config file

"""
from copy import deepcopy

from torch.utils.data import DataLoader

from modules.helpers.csv_saver import CSVSaver
from modules.helpers.namer import Namer
from utils.constants import TRAIN_RESULTS_DIR
from utils.flags import train_flags
from utils.common import build_config, setup_imports, setup_directories, add_grid_search_parameters
from utils.registry import registry
import pandas as pd
from ray import tune
from typing import Dict, List


def train_one(configs: Dict, save: bool = False) -> None:
    """ Prepares Dataset """
    setup_imports()
    dls = get_data_loaders(configs)
    trainer = registry.get_trainer_class(
        configs.get('trainer').get('name')
    )(configs, dls)
    trainer.train()
    if save:
        trainer.save()


def get_data_loaders(configs):
    dls = []
    for name in ['train', 'valid', 'test']:
        hps = configs.get('dataset').get('data_loaders').get(name)
        dl = get_data_loader(configs, name, hps)
        dls.append(dl)

    return dls


def get_data_loader(configs, name, hps):
    dataset = registry.get_dataset_class(configs.get('dataset').get('name'))(configs, name)
    return DataLoader(dataset, **hps, collate_fn=dataset.collate)


def train(configs: Dict) -> None:
    grid = add_grid_search_parameters(configs)
    sync_config = tune.SyncConfig(sync_to_driver=False)
    config_copy = deepcopy(configs)
    if grid:
        analysis = tune.run(
            train_one,
            config=configs,
            local_dir=TRAIN_RESULTS_DIR,
            sync_config=sync_config,
            name=Namer.wrapper_name(configs.get('model')),
            **config_copy.get('trainer').get('tune', {}),
            keep_checkpoints_num=1,
            checkpoint_score_attr=configs.get('trainer').get('grid_metric').get('name'),
            mode=configs.get('trainer').get('grid_metric').get('mode'),
        )
        best_configs = analysis.get_best_config(
            metric=configs.get('trainer').get('grid_metric').get('name'),
            mode=configs.get('trainer').get('grid_metric').get('mode')
        )
        print("Best configs: ", {**best_configs.get('optim'),
                                 **best_configs.get('special_inputs')})
        configs = best_configs
    train_one(configs, save=configs.get('trainer').get('save'))


if __name__ == '__main__':
    setup_imports()
    setup_directories()
    parser = train_flags.parser
    args = parser.parse_args()
    config = build_config(args)
    train(config)
