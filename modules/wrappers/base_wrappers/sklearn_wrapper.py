from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper


class SKLearnWrapper(DefaultWrapper):
    """ Any neural net model in pytorch """

    def predict_proba(self, examples):
        """ makes prediction on pandas examples of dim N X M
                 where N is number of examples and M number of features """
        if self._features_list:
            examples = examples[self._features_list]
        else:
            examples = examples.iloc[:, 2:]
        examples = examples.values.astype(float)
        if len(examples.shape) == 1:
            examples = examples.reshape((1, -1))

        return (
                self.clf.predict_proba(
                    examples
                )
            )

    def fit(self, inputs, targets):
        self.clf.fit(inputs, targets)

    def forward(self, examples):
        """ outputs probs, rewrite if your sklearn model
                                    doesnt have this function"""
        return self.clf.predict_proba(examples)

