from abc import ABC, abstractmethod
from modules.interfaces.saver import Saver
from utils.registry import registry
from modules.containers.container import Container
import pandas as pd


class CSVSaver(Saver):

    def save(self, data):
        """ saves csv to output_csv which is local path """
        if isinstance(data, pd.DataFrame):
            data.to_csv(Container.config.get('dataset').get('output_path'))



