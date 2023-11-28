import numpy as np
from sklearn.preprocessing import add_dummy_feature
from ..loss.loss_factory import LossFactory
from ..regularization.reg_factory import RegularizationFactory
from ..learning_schedule.lr_factory import LearningScheduleFactory

class LinearRegressionSGD:
    """
        This class implements the Linear Regression with Stochastic Gradient Descent
    """

    def __init__(self, epochs=1000, eta=0.1, add_bias=True, loss='mse', reg=None, alpha=0.1, lr=None):
        self.epochs = epochs
        self.eta = eta
        self.theta_best = None
        self.add_bias = add_bias
        self.loss = LossFactory.create(loss)
        self.alpha = alpha

        self.reg = RegularizationFactory.create(reg)
        self.lr = LearningScheduleFactory.create(lr)

    def fit(self, X, y):
        m = X.shape[0]
        if self.add_bias:
            X = add_dummy_feature(X)
        y = y.reshape(-1, 1)
        self.theta_best = np.random.randn(X.shape[1], 1)
        for epoch in range(self.epochs):
            for iteration in range(m):
                random_idx = np.random.randint(m)
                xi = X[random_idx:random_idx + 1]
                yi = y[random_idx:random_idx + 1]
                loss_gradient = self.eta * xi.T.dot(self.loss.gradient(yi, xi.dot(self.theta_best)))
                reg_gradient = self.alpha * self.reg.gradient(self.theta_best)
                gradients = loss_gradient + reg_gradient
                self.eta = self.lr.get_learning_rate(self.eta)
                self.theta_best -= gradients

    def predict(self, X):
        if self.theta_best is None:
            raise Exception("Model not trained yet.")
        if self.add_bias:
            X = add_dummy_feature(X)
        return X.dot(self.theta_best)