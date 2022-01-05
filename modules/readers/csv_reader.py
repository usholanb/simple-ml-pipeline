from modules.readers.base_reader.default_reader import DefaultReader
from utils.registry import registry
import pandas as pd
from typing import AnyStr


@registry.register_reader('csv_reader')
class CSVReader(DefaultReader):
    """ Reads CSV files """

    def read_source(self, input_path: AnyStr) -> pd.DataFrame:
        """ reads csv files and returns pandas DataFrame"""
        df = pd.read_csv(f'{input_path}.csv')
        if df.columns[0] == 'Unnamed: 0':
            df = df.drop(columns=[df.columns[0]])
        return df





