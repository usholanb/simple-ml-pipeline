import argparse
from abc import ABC, abstractmethod
from typing import AnyStr
from utils.constants import CONFIGS_DIR


class Flags:
    def __init__(self, description: AnyStr):
        self._parser = self.get_parser(description)
        self.add_core_args()

    @property
    def parser(self) -> argparse.ArgumentParser:
        return self._parser

    @staticmethod
    def get_parser(description='Default Parser Description'):
        """ Default Parser """
        return argparse.ArgumentParser(
            description=description
        )

    @abstractmethod
    def add_core_args(self):
        """ Choose config file for example, use parser.add_argument """


class PreprocessingFlag(Flags):

    def add_core_args(self) -> None:
        self._parser.add_argument_group("Core Arguments")
        self._parser.add_argument(
            "--config-yml",
            default=f'{CONFIGS_DIR}/preprocessing.yml',
            help="path to config file starting from project home path",
        )


class TrainFlags(Flags):

    def add_core_args(self) -> None:
        self._parser.add_argument_group("Core Arguments")
        self._parser.add_argument(
            "--config-yml",
            default=f'{CONFIGS_DIR}/train_rfc.yml',
            help="path to config file starting from project home path",
        )

class PredictionFlags(Flags):

    def add_core_args(self) -> None:
        self._parser.add_argument_group("Core Arguments")
        self._parser.add_argument(
            "--config-yml",
            default=f'{CONFIGS_DIR}/prediction.yml',
            help="path to config file starting from project home path",
        )


preprocessing_flags = PreprocessingFlag('dataset creation config file')
train_flags = TrainFlags('trainer creation config file')
prediction_flags = PredictionFlags('prediction config file')
