import numpy as np
from sklearn.preprocessing import add_dummy_feature

from ..loss.loss_factory import LossFactory
from ..regularization.l1 import L1
from ..learning_schedule.lr_factory import LearningScheduleFactory



class LassoRegression:
    """
        This class implements Lasso Regression using gradient descent.
    """
    def __init__(self, epochs=1000, eta=0.1, add_bias=True, alpha=0.1, loss='mse'):
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.theta_best = None
        self.add_bias = add_bias
        self.reg = L1()
        self.loss = LossFactory.create(loss)
        self.lr = LearningScheduleFactory.create()

    def fit(self, X, y):
        m = X.shape[0]
        if self.alpha:
            X = add_dummy_feature(X)
        self.theta_best = np.random.randn(X.shape[1],1)
        y = y.reshape(-1,1)
        for epoch in range(self.epochs):
            loss_gradient = self.eta * X.T.dot(self.loss.gradient(y, X.dot(self.theta_best)))
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