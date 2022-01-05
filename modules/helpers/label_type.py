from typing import AnyStr


class LabelType:

    @property
    def classification(self) -> bool:
        return self.get_label_type() == 'classification'

    @property
    def regression(self) -> bool:
        return self.get_label_type() == 'regression'

    def get_label_type(self) -> AnyStr:
        conf = self.configs.get('dataset', None)
        if conf is None:
            conf = self.configs.get('reader', None)
        return conf.get('label_type')
