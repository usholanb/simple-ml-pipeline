import logging
import sys
import importlib.util as importlib_util
import yaml
from typing import List, AnyStr, Dict
from pathlib import Path
import copy
import os
import importlib
import glob
from typing import AnyStr
import pickle
import re
import ray


from utils.constants import DATA_DIR, CONFIGS_DIR, PREDICTIONS_DIR, TRAIN_RESULTS_DIR, PROJECT_DIR
import numpy as np


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

    UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
    MODULES_DIR = os.path.join(UTILS_DIR, "../modules")

    trainer_folder = os.path.join(MODULES_DIR, "trainers")
    trainer_pattern = os.path.join(trainer_folder, "**", "*.py")
    datasets_folder = os.path.join(MODULES_DIR, "datasets")
    datasets_pattern = os.path.join(datasets_folder, "*.py")
    model_folder = os.path.join(MODULES_DIR, "models")
    model_pattern = os.path.join(model_folder, "*.py")
    wrapper_folder = os.path.join(MODULES_DIR, "wrappers")
    wrapper_pattern = os.path.join(wrapper_folder, "*.py")
    transformer_folder = os.path.join(MODULES_DIR, "transformers")
    transformer_pattern = os.path.join(transformer_folder, "*.py")
    metric_folder = os.path.join(MODULES_DIR, "metrics")
    metric_pattern = os.path.join(metric_folder, "*.py")
    loss_folder = os.path.join(MODULES_DIR, "losses")
    loss_pattern = os.path.join(loss_folder, "*.py")


    # importlib.import_module("utils.common.logger")

    files = (
        glob.glob(datasets_pattern, recursive=True)
        + glob.glob(model_pattern, recursive=True)
        + glob.glob(trainer_pattern, recursive=True)
        + glob.glob(wrapper_pattern, recursive=True)
        + glob.glob(transformer_pattern, recursive=True)
        + glob.glob(metric_pattern, recursive=True)
        + glob.glob(loss_pattern, recursive=True)

    )

    for f in files:
        for key in ["/trainers", "/datasets", "/models", "/wrappers",
                    "/transformers", "/metrics", "/losses"]:
            if f.find(key) != -1:
                splits = f.split(os.sep)
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


def create_folder(folder_path: AnyStr) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def setup_directories() -> None:
    for folder_path in dir(sys.modules['utils.constants']):
        if folder_path.endswith('_DIR'):
            create_folder(getattr(sys.modules['utils.constants'], folder_path))


def pickle_obj(obj, path: AnyStr) -> None:
    with open(path, 'wb') as f_in:
        pickle.dump(obj, f_in)


def unpickle_obj(path: AnyStr):
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
