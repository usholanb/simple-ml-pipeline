import torch

from modules.trainers.base_trainer import BaseTrainer
import sklearn
from typing import AnyStr
import numpy as np


class DefaultTrainer(BaseTrainer):
    def __init__(self, configs, dataset):
        self.configs = configs
        self.dataset = dataset

