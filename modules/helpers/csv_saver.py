import pandas as pd
from typing import AnyStr


class CSVSaver:

    def __init__(self, config):
        self.output_path = config.get('dataset').get('output_path')

    def save(self, data, config) -> None:
        """ saves csv to output_csv which is local path """
        if isinstance(data, pd.DataFrame):
            data.to_csv(self.output_path, index=False)
        elif isinstance(data, dict):
            for split, input_path in config.get('dataset').get('input_path').items():
                self.save_file(self.raw_to_processed(input_path), data[split])

    @classmethod
    def raw_to_processed(cls, path: AnyStr) -> AnyStr:
        return f'processed_{path}'

    @staticmethod
    def save_file(path, df):
        df.to_csv(path, index=False)



