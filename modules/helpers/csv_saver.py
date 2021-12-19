import pandas as pd
from typing import AnyStr


class CSVSaver:

    @classmethod
    def save(cls, data, configs) -> None:
        """ saves csv to processed_ + input_path which is local path """
        input_path = configs.get('dataset').get('input_path')
        if isinstance(data, pd.DataFrame):
            input_path = input_path.rstrip('.csv')
            k_fold_tag = configs.get('dataset').get('k_fold_tag', '')
            name = f"{cls.raw_to_processed(f'{input_path}{k_fold_tag}')}"
            data.to_csv(name, index=False, compression='gzip')

    @classmethod
    def load(cls, configs) -> pd.DataFrame:
        """ loads csv from input_path which is local path"""
        tag = configs.get('dataset').get('k_fold_tag', '')
        input_path = f"{configs.get('dataset').get('input_path')}{tag}"
        gz_input_path = cls.add_csv_gz(input_path)
        dataset = pd.read_csv(gz_input_path,  compression='gzip')
        print(f'reading {gz_input_path}, dataset: {dataset.shape}')
        return dataset

    @classmethod
    def add_csv(cls, path):
        return f'{path}.csv'

    @classmethod
    def add_csv_gz(cls, path):
        path = cls.add_csv(path)
        return f'{path}.gz'

    @classmethod
    def raw_to_processed(cls, path: AnyStr) -> AnyStr:
        return cls.add_csv_gz(f'processed_{path}')

    @classmethod
    def save_file(cls, path, df, compression='gzip', index=False):
        if compression == 'gzip':
            path = cls.add_csv_gz(path)
        else:
            path = cls.add_csv(path)
        df.to_csv(path, index=index,  compression=compression)
        print(f'saved {path}')


