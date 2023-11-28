import numpy as np
from sklearn.preprocessing import add_dummy_feature
from sklearn.preprocessing import OneHotEncoder

from ..base.softmax import Softmax
from ..loss.loss_factory import LossFactory
from ..regularization.reg_factory import RegularizationFactory
from ..learning_schedule.lr_factory import LearningScheduleFactory

class SoftmaxRegression:
    def __init__(self, epochs=1000, add_bias=True, eta=0.1, lr='simple', loss='log_loss', reg='l2', alpha=0.1):
        self.add_bias = add_bias
        self.theta_best = None
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha

        self.lr = LearningScheduleFactory.create(lr)
        self.reg = RegularizationFactory.create(reg)
        self.loss = LossFactory.create(loss)

    def fit(self, X, y):
        if self.add_bias:
            X = add_dummy_feature(X)
        num_classes = len(np.unique(y))
        encoder = OneHotEncoder()
        y = encoder.fit_transform(y.reshape(-1, 1)).toarray()
        self.theta_best = np.random.randn(X.shape[1], num_classes)
        for epoch in range(self.epochs):
            scores = Softmax.forward(X.dot(self.theta_best))
            loss_gradient = self.eta * X.T.dot(self.loss.gradient(y, scores))
            reg_gradient = self.alpha * self.reg.gradient(self.theta_best)
            gradient = loss_gradient + reg_gradient
            self.theta_best -= gradient

    def predict_proba(self, X):
        if self.theta_best is None:
            raise Exception("Model not trained yet.")
        if self.add_bias:
            X = add_dummy_feature(X)
        return Softmax.forward(X.dot(self.theta_best))

    def predict(self, X):
        if self.theta_best is None:
            raise Exception("Model not trained yet.")
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)