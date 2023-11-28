import numpy as np
from sklearn.preprocessing import add_dummy_feature
from pymlscratch.loss.loss_factory import LossFactory
from pymlscratch.regularization.reg_factory import RegularizationFactory

class LinearRegressionGD:
    """
        This class implements the Linear Regression with Gradient Descent
    """
    def __init__(self, epochs=1000, eta=0.1, add_bias=True, reg=None, loss='mse', alpha=0.1, batch=None):
        self.epochs = epochs
        self.eta = eta
        self.theta_best = None
        self.add_bias = add_bias
        self.alpha = alpha
        self.batch = batch


        self.loss = LossFactory.create(loss)
        self.reg = RegularizationFactory.create(reg)

    def fit(self, X, y):
        m = X.shape[0]
        if self.add_bias:
            X = add_dummy_feature(X)
        y = y.reshape(-1, 1)
        self.theta_best = np.random.randn(X.shape[1], 1)
        if self.batch is None:
            self.batch = X.shape[0]

        for epoch in range(self.epochs):
            for iteration in range((m + self.batch - 1) // self.batch):
                random_indices = np.random.randint(m, size=self.batch)
                Xi = X[random_indices]
                yi = y[random_indices]
                loss_gradient = self.eta * Xi.T.dot(self.loss.gradient(yi, Xi.dot(self.theta_best)))
                reg_gradient = self.alpha * self.reg.gradient(self.theta_best)
                gradients = loss_gradient + reg_gradient
                self.theta_best -= gradients

    def predict(self, X):
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
    lin_reg = LinearRegressionGD()
    lin_reg.fit(X, y)
    print(f'Theta Best: {lin_reg.theta_best}')
    print(lin_reg.predict(np.array([[0], [2]])))