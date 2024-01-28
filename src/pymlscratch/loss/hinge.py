import numpy as np

class HingeLoss:
    """
    This class implements the hinge loss function.
    """
    @classmethod
    def loss(cls, y_true, y_pred):
        return np.maximum(0, 1 - y_true * y_pred)

    @classmethod
    def gradient(cls, y_true, y_pred):
        loss = np.where(cls.loss(y_true, y_pred) > 0, 1, 0)
        return -loss * y_true