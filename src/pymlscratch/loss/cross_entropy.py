import numpy as np


class CrossEntropy:
    @classmethod
    def loss(cls, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred))

    @classmethod
    def gradient(cls, y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]