from modules.datasets.base_dataset import BaseDataset
from modules.models.sklearn_model import SKLearnModel
from modules.trainers.base_trainer import BaseTrainer


class SKLearnTrainer(BaseTrainer):

    def train(self, dataset, model: SKLearnModel):
        """ trains sklearn model with dataset """

