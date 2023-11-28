import numpy as np
from sklearn.preprocessing import add_dummy_feature

class LinearRegression:
    """
        This class implements Linear Regression using the Normal equation.
    """
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        self.theta_best = None

    def fit(self, X, y):
        """
            This function fits the model to the training data.
        """
        if self.add_bias:
            X = add_dummy_feature(X)
        self.theta_best = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        """
            This function predicts the output for the given input.
        """
        if self.theta_best is None:
            raise Exception("Model not trained yet.")
        if self.add_bias:
            X = add_dummy_feature(X)
        return X.dot(self.theta_best)


if __name__ == '__main__':
    np.random.seed(42)  # to make this code example reproducible
    m = 100  # number of instances
    X = 2 * np.random.rand(m, 1)  # column vector
    y = 4 + 3 * X + np.random.randn(m, 1)  # column vector
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(lin_reg.predict(np.array([[0], [2]])))