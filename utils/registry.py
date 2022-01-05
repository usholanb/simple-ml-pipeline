"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Copyright (c) Facebook, Inc. and its affiliates.
# Borrowed from https://github.com/facebookresearch/pythia/blob/master/pythia/common/registry.py.
from typing import AnyStr, Callable

from utils.constants import FOLDERS_NAMES

"""
Registry is central source of truth. Inspired from Redux's concept of
global store, Registry maintains mappings of various information to unique
keys. Special functions in registry can be used as decorators to register
different kind of classes.
Import the global registry object using
``from ocpmodels.common.registry import registry``
Various decorators for registry different kind of classes with unique keys
- Register a model: ``@registry.register_model``
"""


class Registry:
    r"""Class for registry object which acts as central source of truth."""
    # Mappings to respective classes.
    mapping = {
        **dict([(f"{name}_name_mapping", {}) for name in FOLDERS_NAMES]),
        "state": {},
    }

    @classmethod
    def add_this_to(cls, func: Callable, name: AnyStr, mapping_name: AnyStr) -> None:
        if name not in cls.mapping[mapping_name]:
            cls.mapping[mapping_name][name] = func

    @classmethod
    def register_dataset(cls, name: AnyStr) -> Callable:
        def wrap(func: Callable) -> Callable:
            cls.add_this_to(func, name, 'datasets_name_mapping')
            return func
        return wrap

    @classmethod
    def register_reader(cls, name: AnyStr) -> Callable:
        def wrap(func: Callable) -> Callable:
            cls.add_this_to(func, name, 'readers_name_mapping')
            return func
        return wrap

    @classmethod
    def register_wrapper(cls, name: AnyStr) -> Callable:

        def wrap(func: Callable) -> Callable:
            cls.add_this_to(func, name, 'wrappers_name_mapping')
            return func

        return wrap

    @classmethod
    def register_transformer(cls, name: AnyStr) -> Callable:

        def wrap(func: Callable) -> Callable:
            cls.add_this_to(func, name, 'transformers_name_mapping')
            return func

        return wrap

    @classmethod
    def register_metric(cls, name: AnyStr) -> Callable:

        def wrap(func: Callable) -> Callable:
            cls.add_this_to(func, name, 'metrics_name_mapping')
            return func

        return wrap

    @classmethod
    def register_model(cls, name: AnyStr) -> Callable:

        def wrap(func: Callable) -> Callable:
            cls.add_this_to(func, name, 'models_name_mapping')
            return func

        return wrap

    @classmethod
    def register_loss(cls, name: AnyStr) -> Callable:
        def wrap(func: Callable) -> Callable:
            cls.add_this_to(func, name, 'losses_name_mapping')
            return func

        return wrap

    @classmethod
    def register_trainer(cls, name: AnyStr) -> Callable:
        def wrap(func: Callable) -> Callable:
            cls.add_this_to(func, name, 'trainers_name_mapping')
            return func

        return wrap

    @classmethod
    def register_predictor(cls, name: AnyStr) -> Callable:

        def wrap(func: Callable) -> Callable:
            cls.add_this_to(func, name, 'predictors_name_mapping')
            return func

        return wrap

    @classmethod
    def register(cls, name: AnyStr, obj) -> None:
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def get_dataset_class(cls, name: AnyStr) -> Callable:
        return cls.mapping["datasets_name_mapping"].get(name, None)

    @classmethod
    def get_reader_class(cls, name: AnyStr) -> Callable:
        return cls.mapping["readers_name_mapping"].get(name, None)

    @classmethod
    def get_wrapper_class(cls, name: AnyStr) -> Callable:
        return cls.mapping["wrappers_name_mapping"].get(name, None)

    @classmethod
    def get_transformer_class(cls, name: AnyStr) -> Callable:
        return cls.mapping["transformers_name_mapping"].get(name, None)

    @classmethod
    def get_metric_class(cls, name: AnyStr) -> Callable:
        return cls.mapping["metrics_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name: AnyStr) -> Callable:
        return cls.mapping["models_name_mapping"].get(name, None)

    @classmethod
    def get_loss_class(cls, name: AnyStr) -> Callable:
        return cls.mapping["losses_name_mapping"].get(name, None)

    @classmethod
    def get_logger_class(cls, name: AnyStr) -> Callable:
        return cls.mapping["loggers_name_mapping"].get(name, None)

    @classmethod
    def get_trainer_class(cls, name: AnyStr) -> Callable:
        return cls.mapping["trainers_name_mapping"].get(name, None)

    @classmethod
    def get_predictor_class(cls, name: AnyStr) -> Callable:
        return cls.mapping["predictors_name_mapping"].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning: bool =False):
        r"""Get an item from registry with key 'name'
        Args:
            name (string): Key whose value needs to be retreived.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for cgcnn's
                               internal operations. Default: False
        Usage::
            from ocpmodels.common.registry import registry
            config = registry.get("config")
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].write(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name: AnyStr) -> Callable:
        r"""Remove an item from registry with key 'name'
        Args:
            name: Key which needs to be removed.
        Usage::
            from ocpmodels.common.registry import registry
            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()
