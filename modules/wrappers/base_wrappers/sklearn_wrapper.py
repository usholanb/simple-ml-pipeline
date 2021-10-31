from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper


class SKLearnWrapper(DefaultWrapper):
    """ Any neural net model in pytorch """

    def predict(self, examples):
        """ makes prediction on pandas examples of dim N X M
                 where N is number of examples and M number of features """
        return (
                self.clf.predict(
                    examples[self.config.get('features_list')].values.astype(float)
                )
            )

    def fit(self, inputs, targets):
        self.clf.fit(inputs, targets)

    def forward(self, examples):
        """ outputs probs, rewrite if your sklearn model
                                    doesnt have this function"""
        return self.clf.predict_proba(examples)

