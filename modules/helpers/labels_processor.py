from typing import AnyStr, Dict, Tuple
import pandas as pd
from modules.helpers.label_type import LabelType


class LabelsProcessor(LabelType):
    def __init__(self, configs: Dict):
        self.configs = configs
        sc = self.configs.get('static_columns')
        self.label_i = sc.get('FINAL_LABEL_NAME_INDEX')
        self.label_index_i = sc.get('FINAL_LABEL_INDEX')

    def get_label_name(self, reader_configs: Dict, data: pd.DataFrame) -> AnyStr:
        target = reader_configs.get('label')
        if isinstance(target, str):
            label_name = target
        elif isinstance(target, int):
            label_name = data.columns[target]
        else:
            raise TypeError('label must be int or str')
        return label_name

    def process_labels(self, reader_configs: Dict, data: pd.DataFrame) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        data is all data
        return: restructured data, if classification label index is added
        """
        label_name = self.get_label_name(reader_configs, data)

        y_end_index = len(self.configs.get('static_columns'))
        if self.classification:
            label_types = data[label_name].unique()
            label_types = dict([(v, k) for (k, v) in enumerate(sorted(label_types))])
            labels = data.iloc[:, data.columns.tolist().index(label_name)].tolist()
            labels_idx = [label_types[l] for l in labels]
            data.insert(loc=self.label_index_i, column=f'{label_name}_index', value=labels_idx)
            if isinstance(data.iloc[0, self.label_index_i], float):
                data.iloc[:, self.label_index_i] = data.iloc[:, self.label_index_i].astype('int32')
            data = LabelsProcessor.reset_label_index(data, label_name, self.label_i)
        elif self.regression:
            data = LabelsProcessor.reset_label_index(data, label_name, self.label_index_i)
            data.insert(loc=self.label_i, column='dummy column', value=None)
        else:
            raise ValueError('please set reader.label_type in configs '
                             'to classification or regression')
        data_x = data.iloc[:, y_end_index:]
        data_y = data.iloc[:, :y_end_index]
        return data_x, data_y

    @staticmethod
    def reset_label_index(data: pd.DataFrame, label_index: (AnyStr, int),
                          result_index) -> pd.DataFrame:
        columns = data.columns.tolist()
        if isinstance(label_index, str):
            label_index = columns.index(label_index)
        column_for_label = columns[result_index]
        columns[result_index] = columns[label_index]
        columns[label_index] = column_for_label
        return data[columns]
