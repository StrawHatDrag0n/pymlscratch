import numpy as np
from sklearn.preprocessing import add_dummy_feature

class RidgeRegression:
    """
        This class implements Ridge Regression using the Normal equation.
    """
    def __init__(self, add_bias=True, alpha=0.1):
        self.add_bias = add_bias
        self.theta_best = None
        self.alpha = alpha

    def fit(self, X, y):
        if self.add_bias:
            X = add_dummy_feature(X)
        self.theta_best = np.linalg.pinv(X.T.dot(X) + self.alpha * np.eye(X.shape[1])).dot(X.T).dot(y)

    def predict(self, X):
        if self.theta_best is None:
            raise Exception("Model not trained yet.")
        if self.add_bias:
            X = add_dummy_feature(X)
        return X.dot(self.theta_best)