from dependency_injector.wiring import Provide

from modules.containers.preprocessing_container import PreprocessingContainer
from modules.datasets.base_dataset import BaseDataset
from modules.interfaces.saver import Saver
from utils.registry import registry
import pandas as pd


@registry.register_model('toy')
class Toy(BaseDataset):
    """ Footlball specific dataset """

    def collect(self):
        df = pd.read_csv(self.config.get('dataset').get('source'))
        print(df.values[0])
        self.data = df

    def save(self, saver: Saver = Provide[PreprocessingContainer.csv_saver]):
        super().save(saver)
