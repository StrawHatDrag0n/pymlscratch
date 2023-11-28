import numpy as np


class MSE:
    """
        This class implements the Mean Squared Error
    """

    @classmethod
    def loss(cls, y_true, y_pred) -> float:
        return np.mean((y_pred - y_true) ** 2)

    @classmethod
    def gradient(cls, y_true, y_pred) -> float:
        return (y_pred - y_true) / y_true.shape[0]
