import numpy as np


class LogLoss:
    @classmethod
    def loss(cls, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @classmethod
    def gradient(cls, y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]