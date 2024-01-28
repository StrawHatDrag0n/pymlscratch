import numpy as np


class MAE:
    @classmethod
    def score(cls, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))