import numpy as np
from sklearn.preprocessing import add_dummy_feature
from ..base.sigmoid import Sigmoid
from ..loss.loss_factory import LossFactory
from ..learning_schedule.lr_factory import LearningScheduleFactory
from ..regularization.reg_factory import RegularizationFactory


class LogisticRegression:
    def __init__(self, add_bias = True, epochs = 10000, eta = 0.1, lr = 'simple', loss = 'log_loss', reg='l2', C=1):
        self.theta_best = None
        self.add_bias = add_bias
        self.epochs = epochs
        self.eta = eta
        self.alpha = 1 / C
        self.lr = LearningScheduleFactory.create(lr)
        self.reg = RegularizationFactory.create(reg)
        self.loss = LossFactory.create(loss)

    def fit(self, X, y):
        if self.add_bias:
            X = add_dummy_feature(X)
        self.theta_best = np.random.randn(X.shape[1], 1)
        y = y.reshape(-1, 1)
        for epoch in range(self.epochs):
            y_pred = Sigmoid.forward(X.dot(self.theta_best))
            loss_gradient = self.eta * X.T.dot(self.loss.gradient(y, y_pred))
            reg_gradient = self.alpha * self.reg.gradient(self.theta_best)
            gradient = loss_gradient + reg_gradient
            self.theta_best -= gradient

    def predict_probs(self, X):
        if self.theta_best is None:
            raise Exception("Model not trained yet.")
        if self.add_bias:
            X = add_dummy_feature(X)
        probs = np.zeros((X.shape[0], 2))
        probs[:, 1] = Sigmoid.forward(X.dot(self.theta_best)).flatten()
        probs[:, 0] = 1 - probs[:, 1]
        return probs


    def predict(self, X):
        if self.theta_best is None:
            raise Exception("Model not trained yet.")
        probs = self.predict_probs(X)
        return probs[:, 1] >= 0.5