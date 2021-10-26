from abc import ABC, abstractmethod
from modules.helpers.saver import Saver
from utils.registry import registry
import pandas as pd


class CSVSaver(Saver):

    def __init__(self, config):
        self.output_path = config.get('dataset').get('output_path')

    def save(self, data, config):
        """ saves csv to output_csv which is local path """
        if isinstance(data, pd.DataFrame):
            data.to_csv(self.output_path)



