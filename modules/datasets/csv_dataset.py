from modules.datasets.base_datasets.default_dataset import DefaultDataset
from utils.registry import registry
import pandas as pd
from typing import AnyStr


@registry.register_dataset('csv_dataset')
class CSVDataset(DefaultDataset):
    """ Reads CSV files """

    def read_source(self, input_path: AnyStr) -> pd.DataFrame:
        """ reads csv files and returns pandas DataFrame"""
        df = pd.read_csv(input_path)
        if df.columns[0] == 'Unnamed: 0':
            df = df.drop(columns=[df.columns[0]])
        return df



