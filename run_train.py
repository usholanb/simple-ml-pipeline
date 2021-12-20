"""
this scripts makes one (useful for debugging) or multiple
training of the same model with different hyper-parameters

The best model can be saved

All details of the training are specified in the config file

"""
from copy import deepcopy

from modules.helpers.csv_saver import CSVSaver
from modules.helpers.namer import Namer
from utils.constants import TRAIN_RESULTS_DIR
from utils.common import build_config, setup_imports, setup_directories, add_grid_search_parameters
from utils.registry import registry
import pandas as pd
from ray import tune
from typing import Dict, AnyStr
import time


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
    print('remove this line in train func in run_train!!!!!!!!!!!!!!')
    dataset = dataset[dataset['value'] > 6]
    print(f'dataset size : {dataset.shape}')
    grid = add_grid_search_parameters(configs)
    sync_config = tune.SyncConfig(sync_to_driver=False)
    config_copy = deepcopy(configs)
    if grid:
        analysis = tune.run(
            tune.with_parameters(train_one, dataset=dataset),
            config=configs,
            local_dir=TRAIN_RESULTS_DIR,
            sync_config=sync_config,
            name=Namer.wrapper_name(configs.get('model')),
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


def run_train(k_fold_tag: AnyStr = ''):
    print('started run_train')
    start = time.time()
    setup_directories()
    from utils.flags import all_flags
    parser = all_flags['train'].parser
    args = parser.parse_args()
    configs = build_config(args)
    configs['dataset']['k_fold_tag'] = k_fold_tag
    train(configs)
    print(f'training took {round(time.time() - start, 2)} seconds')


if __name__ == '__main__':
    run_train()
