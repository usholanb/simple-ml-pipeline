from typing import Dict, Tuple
from abc import ABC, abstractmethod
import pandas as pd
import torch


class BaseModel(ABC):

    @abstractmethod
    def predict(self, data: (Dict, pd.DataFrame)):
        """
        Used during prediction step
        Forward + optional transformations for predictions creation
        """

    @abstractmethod
    def get_x_y(self, batch) -> Tuple:
        """ splits original Dataloader batch to x and y,
        where x.shape[0] == y.shape[0] is the number of examples

        Returns (x, y) where
        x - anything(typically tensor or list of tensors, doesn't contain y)
        y - tensor
        """

    @abstractmethod
    def forward(self, data: Dict) -> torch.Tensor:
        """
        data: contains x - the input, batch_size, epoch, iteration index

        returns: outputs of size [batch_size x n_outputs]
        """

    @abstractmethod
    def add_hooks(self) -> None:
        """ add hooks before and after main trainer functions

            For reference: look at your trainer functions
                that are decorated with hooks
         """
    
    @abstractmethod
    def model_epoch_logs(self) -> Dict:
        """ Return anything that needs to be logged in tensorboard
            each epoch
        """
