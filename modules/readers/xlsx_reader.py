from modules.readers.base_reader.default_reader import DefaultReader
from utils.registry import registry
import pandas as pd
from typing import AnyStr


@registry.register_reader('xlsx_reader')
class XLSXReader(DefaultReader):
    """ Reads XLSX files """

    def read_source(self, input_path: AnyStr) -> pd.DataFrame:
        """ reads xlsx files and returns pandas DataFrame"""
        df = pd.read_excel(f'{input_path}.xlsx', sheet_name=
            self.configs.get('reader').get('sheet_name', None))
        if df.columns[0] == 'Unnamed: 0':
            df = df.drop(columns=[df.columns[0]])
        return df



