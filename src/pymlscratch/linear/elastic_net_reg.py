import numpy as np
from sklearn.preprocessing import add_dummy_feature
from ..regularization.reg_factory import RegularizationFactory
from ..learning_schedule.lr_factory import LearningScheduleFactory
from ..loss.loss_factory import LossFactory


class ElasticNetRegression:
    def __init__(self, epochs=10000, eta=0.1, add_bias=True, alpha=1.0, l1_ratio=0.5, lr='simple', loss='mse'):
        self.alpha = alpha
        self.epochs = epochs
        self.eta = eta
        self.l1_ratio = l1_ratio
        self.theta_best = None
        self.add_bias = add_bias
        self.loss = LossFactory.create(loss)
        self.lr = LearningScheduleFactory.create(lr)
        self.reg = RegularizationFactory.create('elastic_net')

    def fit(self, X, y):
        if self.add_bias:
            X = add_dummy_feature(X)

        self.theta_best = np.random.randn(X.shape[1], 1)
        y = y.reshape(-1, 1)
        for epoch in range(self.epochs):
            loss_gradient = self.eta * X.T.dot(self.loss.gradient(y, X.dot(self.theta_best)))
            reg_gradient = self.alpha * self.reg.gradient(self.theta_best)
            gradient = loss_gradient + reg_gradient
            self.theta_best -= gradient

    def predict(self, X):
        if self.theta_best is None:
            raise Exception("Model not trained yet.")
        if self.add_bias:
            X = add_dummy_feature(X)
        return X.dot(self.theta_best)