from modules.containers.di_containers import SaverContainer
from modules.datasets.default_dataset import DefaultDataset
from modules.helpers.saver import Saver
from utils.registry import registry
import pandas as pd
from typing import Dict, AnyStr, List


@registry.register_dataset('csv_dataset')
class CSVDataset(DefaultDataset):
    """ Reads CSV files """

    def read_source(self, input_path: AnyStr) -> pd.DataFrame:
        """ reads csv files and returns pandas DataFrame"""
        df = pd.read_csv(input_path)
        if df.columns[0] == 'Unnamed: 0':
            df = df.drop(columns=[df.columns[0]])
        return df

    def shuffle(self, data: pd.DataFrame):
        return data.sample(frac=1).reset_index(drop=True)

    def save(self, saver: Saver) -> None:
        saver.save(self.data, self.config)

    def reset_label_index(self, data: Dict, index: (AnyStr, int)) -> Dict:
        for split_name, split in data.items():
            columns = split.columns.tolist()
            if isinstance(index, str):
                index = columns.index(index)
            first_column = columns[0]
            columns[0] = columns[index]
            columns[index] = first_column
            data[split_name] = split[columns]
        return data


