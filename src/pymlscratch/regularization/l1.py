import numpy as np

class L1:
    """
        This class implements the L1 Regularization
    """
    def __init__(self, *args, **kwargs):
        pass
    def regularize(self, theta):
        return np.sum(np.abs(theta))

    def gradient(self, theta):
        return np.sign(theta)
