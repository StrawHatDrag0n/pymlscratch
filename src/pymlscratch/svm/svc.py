import numpy as np
from sklearn.preprocessing import add_dummy_feature
from ..loss.loss_factory import LossFactory
from ..regularization.reg_factory import RegularizationFactory
from ..learning_schedule.lr_factory import LearningScheduleFactory
class SVC:
    def __init__(self, eta=0.01, C=1.0, epoch=1000, seed=42):
        self.seed = seed
        self.epoch = epoch
        self.eta = eta
        self.C = C
        self.loss = LossFactory.create('hinge_loss')
        self.reg = RegularizationFactory.create('l2')
        self.w = None
        self.b = 0
        self.lr_schedule = LearningScheduleFactory.create('simple')

        np.random.seed(self.seed)

    def fit(self, X, y):
        self.w = np.random.rand(X.shape[1], 1)
        y = y.reshape(-1, 1)
        y = np.where(y == 0, -1, 1)
        m = X.shape[0]
        for epoch in range(self.epoch):
            for _ in range(m):
                random_index = np.random.randint(m)
                X_i = X[random_index:random_index+1]
                y_i = y[random_index:random_index+1]
                y_pred = X_i.dot(self.w) + self.b
                loss_gradient = self.eta * X_i.T.dot(self.loss.gradient(y_i, y_pred))
                reg_gradient = (1.0 / self.C)  * (self.reg.gradient(self.w) / 2)
                gradient = loss_gradient + reg_gradient
                self.w -= gradient
                self.b -= self.eta * np.sum(self.loss.gradient(y, y_pred))
                self.eta = self.lr_schedule.get_learning_rate(self.eta)

    def predict(self, X):
        X = np.array(X)
        return np.where(np.sign(X.dot(self.w) + self.b) <= 0, False, True).reshape(-1, 1)
