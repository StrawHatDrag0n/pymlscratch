class BaseLinearRegressionMixin:
    def __init__(self, add_bias=True, *args, **kwargs):
        self.add_bias = add_bias
        self.theta_best = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass