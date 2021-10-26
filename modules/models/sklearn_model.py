from modules.models.base_model import BaseModel
from modules.models.default_model import DefaultModel


class SKLearnModel(DefaultModel):
    """ Any neural net model in pytorch """

    def predict(self, examples):
        """ makes prediction on examples of dim N X M where N is number of
          examples and M number of features """
        return self.clf.predict(examples)
