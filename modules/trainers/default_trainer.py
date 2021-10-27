import torch

from modules.trainers.base_trainer import BaseTrainer
import sklearn
from typing import AnyStr
import numpy as np


class DefaultTrainer(BaseTrainer):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

