import pandas as pd
from typing import AnyStr, Dict


class CSVSaver:

    @classmethod
    def save(cls, data: pd.DataFrame, reader_configs: Dict) -> None:
        """ saves csv to processed_ + input_path which is local path """
        path = reader_configs.get('input_path')
        if isinstance(data, pd.DataFrame):
            path = path.split('.csv')[0]
            k_fold_tag = reader_configs.get('k_fold_tag', '')
            name = f"{cls.raw_to_processed(f'{path}{k_fold_tag}')}"
            data.to_csv(name, index=False, compression='gzip')

    @classmethod
    def load(cls, configs: Dict, folder: AnyStr = '') -> pd.DataFrame:
        """ loads csv from input_path which is local path"""
        tag = configs.get('dataset').get('k_fold_tag', '')
        input_path = f"{configs.get('dataset').get('input_path')}{tag}"
        gz_input_path = cls.add_csv_gz(input_path)
        print(f'reading {gz_input_path}')
        if folder:
            gz_input_path = f'{folder}/{gz_input_path}'
        dataset = pd.read_csv(gz_input_path,  compression='gzip')
        print(f'dataset: {dataset.shape}')
        return dataset

    @classmethod
    def add_csv(cls, path: AnyStr) -> AnyStr:
        return f'{path}.csv'

    @classmethod
    def add_csv_gz(cls, path: AnyStr) -> AnyStr:
        path = cls.add_csv(path)
        return f'{path}.gz'

    @classmethod
    def raw_to_processed(cls, path: AnyStr) -> AnyStr:
        return cls.add_csv_gz(f'processed_{path}')

    @classmethod
    def save_file(cls, path: AnyStr, df: pd.DataFrame,
                  gzip: bool = True,
                  index: bool = False) -> None:
        if gzip:
            path = cls.add_csv_gz(path)
        else:
            path = cls.add_csv(path)
        df.to_csv(path, index=index, compression='gzip' if gzip else None)
        print(f'saved {path}')
