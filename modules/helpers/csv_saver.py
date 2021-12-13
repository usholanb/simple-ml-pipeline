import pandas as pd
from typing import AnyStr


class CSVSaver:

    @classmethod
    def save(cls, data, configs) -> None:
        """ saves csv to output_csv which is local path """
        input_path = configs.get('dataset').get('input_path')
        if isinstance(data, pd.DataFrame):
            data.to_csv(cls.raw_to_processed(input_path.rstrip('.csv')),
                        index=False, compression='gzip')

    @classmethod
    def load(cls, configs) -> pd.DataFrame:
        """ saves csv to output_csv which is local path """
        input_path = configs.get('dataset').get('input_path')
        gz_input_path = cls.add_csv_gz(input_path)
        print(f'reading {gz_input_path}')
        return pd.read_csv(gz_input_path,  compression='gzip')

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


