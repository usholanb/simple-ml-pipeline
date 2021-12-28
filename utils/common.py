import logging
import sys
import importlib.util as importlib_util
from ray import tune
import torch
import yaml
from typing import List, Dict, Type
from pathlib import Path
import copy
import os
from datetime import timedelta
import importlib
import glob
from typing import AnyStr
import pickle
import re
import ray
import numpy as np
from time import time
from torch.utils.data import DataLoader
from utils.constants import MODULES_DIR, FOLDERS_NAMES
from utils.registry import registry


def load_config(path: AnyStr, previous_includes: List = None):
    previous_includes = [] if previous_includes is None else previous_includes
    path = Path(path)
    if path in previous_includes:
        raise ValueError(
            f"Cyclic config include detected. {path} included in sequence {previous_includes}."
        )
    previous_includes = previous_includes + [path]
    with open(path, "r") as f_in:
        direct_config = yaml.safe_load(f_in)

    # Load config from included files.
    if "includes" in direct_config:
        includes = direct_config.pop("includes")
    else:
        includes = []
    if not isinstance(includes, list):
        raise AttributeError(
            "Includes must be a list, '{}' provided".format(type(includes))
        )

    config = {}
    duplicates_warning = []
    duplicates_error = []

    for include in includes:
        include_config, inc_dup_warning, inc_dup_error = load_config(
            include, previous_includes
        )
        duplicates_warning += inc_dup_warning
        duplicates_error += inc_dup_error

        # Duplicates between includes causes an error
        config, merge_dup_error = merge_dicts(config, include_config)
        duplicates_error += merge_dup_error

    # Duplicates between included and main file causes warnings
    config, merge_dup_warning = merge_dicts(config, direct_config)
    duplicates_warning += merge_dup_warning

    return config, duplicates_warning, duplicates_error


def build_config(args) -> Dict:
    config, duplicates_warning, duplicates_error = load_config(args.config_yml)
    if len(duplicates_warning) > 0:
        logging.warning(
            f"Overwritten config parameters from included configs "
            f"(non-included parameters take precedence): {duplicates_warning}"
        )
    if len(duplicates_error) > 0:
        raise ValueError(
            f"Conflicting (duplicate) parameters in simultaneously "
            f"included configs: {duplicates_error}"
        )
    return config


# Copied from https://github.com/facebookresearch/mmf/blob/master/mmf/utils/env.py#L89.
def setup_imports() -> None:
    from utils.registry import registry

    # First, check if imports are already setup
    has_already_setup = registry.get("imports_setup", no_warning=True)
    if has_already_setup:
        return
    # Automatically load all of the modules, so that
    # they register with registry
    folders = [os.path.join(MODULES_DIR, f) for f in FOLDERS_NAMES]
    patterns = [os.path.join(f, "**", "*.py") for f in folders]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))

    for f in files:
        for key in [f'/{e}' for e in FOLDERS_NAMES]:
            if f.find(key) != -1:
                splits = f.split(os.sep)
                if splits[-2] == key[1:]:
                    file_name = splits[-1]
                    module_name = file_name[: file_name.find(".py")]
                    importlib.import_module(f"modules.{key[1:]}.{module_name}")

    experimental_folder = os.path.join(MODULES_DIR, "../experimental/")
    if os.path.exists(experimental_folder):
        experimental_files = glob.glob(
            experimental_folder + "**/*py",
            recursive=True,
        )
        # Ignore certain directories within experimental
        ignore_file = os.path.join(experimental_folder, ".ignore")
        if os.path.exists(ignore_file):
            ignored = []
            with open(ignore_file) as f:
                for line in f.read().splitlines():
                    ignored += glob.glob(
                        experimental_folder + line + "/**/*py", recursive=True
                    )
            for f in ignored:
                experimental_files.remove(f)
        for f in experimental_files:
            splits = f.split(os.sep)
            file_name = ".".join(splits[-splits[::-1].index("..") :])
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module(module_name)

    registry.register("imports_setup", True)


def merge_dicts(dict1: Dict, dict2: Dict):
    """Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
    value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py
    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share the same key.
    Returns
    -------
    return_dict: dict
        Merged dictionaries.
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)
    duplicates = []

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k], duplicates_k = merge_dicts(dict1[k], dict2[k])
                duplicates += [f"{k}.{dup}" for dup in duplicates_k]
            else:
                return_dict[k] = dict2[k]
                duplicates.append(k)

    return return_dict, duplicates


def create_folder(folder_path: AnyStr) -> AnyStr:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def setup_directories() -> None:
    for folder_path in dir(sys.modules['utils.constants']):
        if folder_path.endswith('_DIR'):
            create_folder(getattr(sys.modules['utils.constants'], folder_path))


def pickle_obj(obj, path: AnyStr) -> None:
    with open(path, 'wb') as f_in:
        pickle.dump(obj, f_in)


def unpickle_obj(path: AnyStr):
    print(f'unpickling {path}')
    with open(path, 'rb') as f_out:
        obj = pickle.load(f_out)
    return obj


def add_grid_search_parameters(configs: Dict) -> bool:
    from ray import tune
    global grid
    grid = False

    def grid_on_par(par: Dict) -> Dict:
        new_par = {}
        global grid
        for k, v in par.items():
            if isinstance(v, list) and k != 'transformers':
                if isinstance(v[0], list) and len(v) == 1:
                    new_par[k] = v[0]
                else:
                    new_par[k] = tune.grid_search(v)
                    grid = True
            else:
                new_par[k] = v
        return new_par

    for p in ['optim', 'special_inputs']:
        param = configs.get(p, {})
        configs[p] = grid_on_par(param)
    return grid


def inside_tune() -> bool:
    return ray.tune.is_session_enabled()


class Singleton(type):
    """ CAREFULL USING IN MULTITHREADING """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def is_outside_library(model_name):
    module_name = '.'.join(model_name.split('.')[:-1])
    return importlib_util.find_spec(module_name)


def get_outside_library(model_name):
    module_name = '.'.join(model_name.split('.')[:-1])
    module = importlib.import_module(module_name)
    class_name = model_name.split('.')[-1]
    return getattr(module, class_name)


def check_label_type(targets):
    empty_array = np.zeros(len(targets))
    np.mod(targets, 1, out=empty_array)
    mask = (empty_array == 0)
    return mask.all()


def to_snake_case(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()


def to_camel_case(name):
    return ''.join(word.title() for word in name.split('_'))


def do_after_iter(iterable, func):
    for i in iter(iterable):
        func(i)
        return i


def transform(all_data, transformers):
    for t in transformers:
        all_data = t.apply(all_data)
    return all_data


def get_transformers(configs):
    setup_imports()
    ts = configs.get('special_inputs').get('transformers', [])
    ts = ts if isinstance(ts, list) else [ts]
    return [registry.get_transformer_class(t_name)(configs)
            for t_name in ts]


class Timeit:
    """ to compute epoch time """
    original_start = None

    def __init__(self, to_print, iter, iter_n=0, every=1):
        self.iter = iter
        self.every = every
        self.start = None
        self.iter_n = iter_n
        self.to_print = to_print

    def __enter__(self):
        self.start = time()
        Timeit.original_start = Timeit.original_start \
            if Timeit.original_start is not None else self.start

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.iter > 0 and self.iter % self.every == 0:
            now = time()
            iter_time = now - self.start
            expected = str(timedelta(seconds=self.iter_n * iter_time))
            print(f'{self.to_print}:   time: {iter_time},    '
                  f'total training time: {round(now - Timeit.original_start, 2)},'
                  f' expected for all {self.iter_n} iters: {expected}')


def get_data_loaders(configs, specific=None):
    split_names = [specific] if specific is not None else ['train', 'valid', 'test']
    d_loaders = []
    name = configs.get('dataset').get('name')
    dataset_class = registry.get_dataset_class(name)

    for split_name in split_names:
        hps = configs.get('dataset').get('data_loaders', {}).get(split_name, {})
        hps.update({'drop_last': True})
        if dataset_class is not None:
            split = dataset_class(configs, split_name)
            hps.update({'collate_fn': split.collate})
            d_loaders.append(DataLoader(split, **hps))
        else:
            input_path = configs.get('dataset').get('input_path')
            print(f'no dataset class with name {name} is found\n'
                  f'will try to look for file with input_path: {input_path}')
            pandas_dataset_class = registry.get_dataset_class('pandas_dataset')
            split = pandas_dataset_class(configs, split_name)
            d_loaders.append(DataLoader(split, **hps))
    return d_loaders


def prepare_train(configs, dataset, split_names=None) -> Dict:
    """
        splits pandas dataframe to train, test, valid
        returns dict of dataframe
    """
    data = {}
    split_names = ['train', 'valid', 'test'] if split_names is None else split_names
    split_i = configs.get('static_columns').get('FINAL_SPLIT_INDEX')
    label_index_i = configs.get('static_columns').get('FINAL_LABEL_INDEX')
    features_list = configs.get('features_list', [])
    if not features_list:
        print('features_list not specified')
    f_list = figure_feature_list(features_list, dataset.columns.tolist())
    for split in split_names:
        split_column = dataset.iloc[:, split_i]
        data[f'{split}_y'] = \
            dataset.loc[split_column == split].iloc[:, label_index_i]
        if features_list:
            data[f'{split}_x'] = \
                dataset.loc[split_column == split][f_list]
        else:
            data[f'{split}_x'] = \
                dataset.loc[split_column == split].iloc[:, len(configs.get('static_columns')):]
    configs['features_list'] = f_list
    return data


def prepare_torch_data(configs, dataset) -> Dict:
    data = prepare_train(configs, dataset)
    configs['special_inputs'].update({'input_dim': data['train_x'].shape[1]})
    torch_data = {}
    classification = configs.get('trainer').get('label_type') == 'classification'
    for split_name, split in data.items():
        if split_name.endswith('_y'):
            split = torch.tensor(split)
            if classification:
                split = split.long()
            else:
                split = split.float()
            torch_data[split_name] = split
        else:
            t = torch.tensor(split)
            torch_data[split_name] = t.float()

    return torch_data


def figure_feature_list(f_list: List, available_features: List) -> List:
    """ Specifying only name of the feature (w/o index for one hot encoded
        features) is enough to allow them in dataset
        f_list: specified in train config file for training,
        available_features: actually inside dataset
        Return: intersection of two lists"""
    final_list = []
    for available_feature in available_features:
        for feature in f_list:
            if feature == available_feature \
                    or '_'.join(available_feature.split('_')[:-1]) == feature:
                final_list.append(available_feature)
    return final_list


def df_type_is(df, dtype: Type) -> bool:
    return (df == df.astype(dtype)).all()


def std(vector, mean):
    sum = torch.zeros(mean.shape).to(device)
    for el in vector:
        sum += el - mean
    return torch.sqrt(torch.abs(sum) / len(vector))


def log_metrics(results) -> None:
    if inside_tune():
        tune.report(**results)
    else:
        to_print = '  '.join([f'{k}: {"{:10.4f}".format(v)}' for k, v in results.items()])
        print(to_print)


def mean_dict_values(ds: List[Dict]) -> Dict:
    """ ds:
            all dicts in ds must have identical keys
        Returns: dict with key-wise mean values of all dict values
    """
    if not ds:
        return {}
    sum_d = {k: 0 for k in ds[0].keys()}
    for d in ds:
        for k, v in d.items():
            sum_d[k] += v
    return {k: v * 1.0 / len(ds) for k, v in sum_d.items()}
