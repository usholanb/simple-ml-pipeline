from modules.datasets.base_dataset import BaseDataset
from modules.models.sklearn_model import SKLearnModel
from modules.trainers.sklearn_trainer import SKLearnTrainer
from utils.registry import registry


@registry.register_model('iris')
class IrisTrainer(SKLearnTrainer):

    def train(self, dataset, model: SKLearnModel):
        """ trains sklearn model with dataset """
        y = dataset['target']
        x = dataset[dataset != 'target']
        print(y[0], x[0])

