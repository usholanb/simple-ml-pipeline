import argparse
from abc import ABC, abstractmethod
from typing import AnyStr
from utils.constants import CONFIGS_DIR, PROJECT_DIR


class Flags:
    def __init__(self, description: AnyStr):
        self._parser = self.get_parser(description)
        self.add_core_args()

    @property
    def parser(self) -> argparse.ArgumentParser:
        return self._parser

    @staticmethod
    def get_parser(description='Default Parser Description')\
            -> argparse.ArgumentParser:
        """ Default Parser """
        return argparse.ArgumentParser(
            description=description
        )

    @abstractmethod
    def add_core_args(self) -> None:
        """ Choose config file for example, use parser.add_argument """


class CustomFlag(Flags):
    def add_config_args(self, config_name) -> Flags:
        self._parser.add_argument_group("Core Arguments")
        self._parser.add_argument(
            "--config-yml",
            default=f'{CONFIGS_DIR}/{config_name}.yml',
            help="path to config file starting from project home path",
        )
        return self


class PreprocessingFlag(Flags):

    def add_core_args(self) -> None:
        self._parser.add_argument_group("Core Arguments")
        self._parser.add_argument(
            "--config-yml",
            default=f'{CONFIGS_DIR}/preprocessing_regression.yml',
            # default=f'{CONFIGS_DIR}/preprocessing.yml',
            # default=f'{PROJECT_DIR}/example_config_files/preprocessing_example.yml',
        )


class TrainFlags(Flags):

    def add_core_args(self) -> None:
        self._parser.add_argument_group("Core Arguments")
        self._parser.add_argument(
            "--config-yml",
            # default=f'{CONFIGS_DIR}/train_dagnet_example.yml',
            # default=f'{CONFIGS_DIR}/train_dense_net_regression.yml',
            # default=f'{PROJECT_DIR}/example_config_files/train_dense_net_example.yml',
            # default=f'{PROJECT_DIR}/example_config_files/train_logistic_regression_example.yml',
            # default=f'{PROJECT_DIR}/example_config_files/train_xgboost_example.yml',
            # default=f'{PROJECT_DIR}/example_config_files/train_dagnet_example.yml',
            default=f'{CONFIGS_DIR}/train_xgboost_regression.yml',
            # default=f'{CONFIGS_DIR}/train_linear_regression.yml',
            # default=f'{CONFIGS_DIR}/train_dense_net.yml',
            help="path to config file starting from project home path",
        )


class PredictionFlags(Flags):

    def add_core_args(self) -> None:
        self._parser.add_argument_group("Core Arguments")
        self._parser.add_argument(
            "--config-yml",
            # default=f'{CONFIGS_DIR}/prediction.yml',
            default=f'{CONFIGS_DIR}/prediction_regression.yml',
            # default=f'{CONFIGS_DIR}/prediction_multi_regression.yml',
            # default=f'{PROJECT_DIR}/example_config_files/prediction_example.yml',
            # default=f'{PROJECT_DIR}/example_config_files/prediction_dagnet_example.yml',
            help="path to config file starting from project home path",
        )


class OrchestraFlags(Flags):

    def add_core_args(self) -> None:
        self._parser.add_argument_group("Core Arguments")
        self._parser.add_argument(
            "--config-yml",
            default=f'{CONFIGS_DIR}/orchestra_regression.yml',
            help="path to config file starting from project home path",
        )


all_flags = {
    'preprocessing': PreprocessingFlag('dataset creation config file'),
    'train': TrainFlags('trainer creation config file'),
    'prediction': PredictionFlags('prediction config file'),
    'orchestra': OrchestraFlags('orchestra config file'),
}

