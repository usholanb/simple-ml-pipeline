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

    def shuffle(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.sample(frac=1).reset_index(drop=True)

    def save(self, saver: Saver) -> None:
        saver.save(self.data, self.configs)

    def reset_label_index(self, data: Dict, label_index: (AnyStr, int)) -> Dict:
        for split_name, split in data.items():
            columns = split.columns.tolist()
            if isinstance(label_index, str):
                label_index = columns.index(label_index)
            label_i = self.configs.get('constants').get('FINAL_LABEL_INDEX')
            column_for_label = columns[label_i]
            columns[label_i] = columns[label_index]
            columns[label_index] = column_for_label
            data[split_name] = split[columns]
        return data


