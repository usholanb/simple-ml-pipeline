from typing import Dict, AnyStr, Tuple
from abc import ABC, abstractmethod
import torch


class BaseModel(ABC):

    @abstractmethod
    def predict(self, data):
        """
        Used during prediction step
        """

    @abstractmethod
    def get_x_y(self, batch) -> Tuple:
        """ splits original Dataloader batch to x and y,
        where x.shape[0] == y.shape[0] is the number of examples

        Returns (x, y) where
        x - tensor or list of tensors, doesn't contain y
        y - tensor
        """

    @abstractmethod
    def forward(self, data: Dict) -> torch.Tensor:
        """
        data: contains x - the input, batch_size, epoch, iteration index

        returns: outputs of size [batch_size x n_outputs]
        """

    @abstractmethod
    def add_hooks(self):
        """ add hooks before and after main trainer functions

            For reference: look at your trainer functions
                that are decorated with hooks
         """
    
    @abstractmethod
    def model_epoch_logs(self) -> Dict:
        """ Return anything that needs to be logged in tensorboard
            each epoch
        """


