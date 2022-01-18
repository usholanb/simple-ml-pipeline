from typing import Dict

from sklearn.preprocessing import MultiLabelBinarizer
from modules.transformers.base_transformers.base_transformer import BaseTransformer
from utils.registry import registry


@registry.register_transformer('mlb')
class MiltiLabelBinarizerTransformer(BaseTransformer):

    def __init__(self, configs: Dict):
        super(MiltiLabelBinarizerTransformer, self).__init__(configs)
        self.mlb = MultiLabelBinarizer()

    def apply(self, data):
        return self.mlb.fit_transform(data)

