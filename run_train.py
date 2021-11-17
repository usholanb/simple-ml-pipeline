"""
this scripts makes one (useful for debugging) or multiple
training of the same model with different hyper-parameters

The best model can be saved

All details of the training are specified in the config file

"""
import os
from copy import deepcopy
from modules.containers.di_containers import TrainerContainer
from modules.helpers.namer import Namer
from utils.constants import TRAIN_RESULTS_DIR, CLASSIFIERS_DIR
from utils.flags import train_flags
from utils.common import build_config, setup_imports, setup_directories, add_grid_search_parameters, get_data_loaders, \
    unpickle_obj
from utils.registry import registry
from ray import tune
from typing import Dict


# from typing import List
#
# from ray import tune
# from ray.tune.checkpoint_manager import Checkpoint
# from ray.tune.trial import Trial
#
#
# class SaveModel(tune.Callback):
#     def on_checkpoint(self, iteration: int, trials: List[Trial],
#                       trial: Trial, checkpoint: Checkpoint, **info):
#         print(type(checkpoint))
#         print(checkpoint)
#         print(checkpoint.__dict__)
#         exit()
#

def get_model(configs):
    if configs.get('trainer', {}).get('resume', False):
        name = Namer.model_name(configs.get('model'))
        folder = CLASSIFIERS_DIR
        model_path = f'{folder}/{name}.pkl'
        if os.path.isfile(model_path):
            model = unpickle_obj(model_path)
            print(f'resumed {name}')
        else:
            raise ValueError(f'cannot resume model {name}'
                             f' - no checkpoint exist in folder {folder}')
    else:
        setup_imports()
        model = registry.get_model_class(
            configs.get('model').get('name')
        )(configs)
    return model


def train_one(configs: Dict, save: bool = False) -> None:
    """ Prepares Dataset """
    setup_imports()
    dls = get_data_loaders(configs)
    model = get_model(configs).to(TrainerContainer.device)
    if configs.get('trainer', {}).get('resume', False):
        configs = model.configs
    trainer = registry.get_trainer_class(
        configs.get('trainer').get('name')
    )(configs, dls, model)
    trainer.train()
    if save:
        trainer.save()


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
            name=Namer.model_name(configs.get('model')),
            **config_copy.get('trainer').get('tune', {}),
            keep_checkpoints_num=1,
            checkpoint_score_attr=configs.get('trainer').get('grid_metric').get('name'),
            mode=configs.get('trainer').get('grid_metric').get('mode'),
            # callbacks=[SaveModel()],
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
