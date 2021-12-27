from modules.readers.base_reader.default_reader import DefaultReader
from utils.registry import registry
import pandas as pd
from typing import AnyStr


@registry.register_reader('xlsx_dataset')
class XLSXReader(DefaultReader):
    """ Reads XLSM files """

    def read_source(self, input_path: AnyStr) -> pd.DataFrame:
        """ reads xlsx files and returns pandas DataFrame"""
        df = pd.read_excel(f'{input_path}.xlsx', sheet_name=
            self.configs.get('dataset').get('sheet_name', 'Sheet1'))
        if df.columns[0] == 'Unnamed: 0':
            df = df.drop(columns=[df.columns[0]])
        return df



