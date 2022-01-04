class LabelType:

    @property
    def classification(self) -> bool:
        return self.configs.get('dataset').get('label_type') == 'classification'

    @property
    def regression(self) -> bool:
        return self.configs.get('dataset').get('label_type') == 'regression'
