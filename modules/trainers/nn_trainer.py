from modules.datasets.base_dataset import BaseDataset
from modules.models.base_model import BaseModel
from modules.trainers.base_trainer import BaseTrainer


class NNTrainer(BaseTrainer):

    def train(self, dataset, model: BaseModel):
        """ trains nn model with dataset """
