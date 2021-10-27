from modules.models.base_models.default_model import DefaultModel


class SKLearnModel(DefaultModel):
    """ Any neural net model in pytorch """

    def predict(self, examples):
        """ makes prediction on examples of dim N X M where N is number of
          examples and M number of features """
        return self.clf.predict(examples)

    def fit(self, inputs, targets):
        self.clf.fit(inputs.to_numpy(), targets.to_numpy())

    def predict_proba(self, examples):
        """ outputs probs, rewrite if your sklearn model
                                    doesnt have this function"""
        return self.clf.predict_proba(examples)

