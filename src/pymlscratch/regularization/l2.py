import numpy as np

class L2:
    """
        This class implements the L2 Regularization
    """
    def __init__(self, *args, **kwargs):
        pass

    def regularize(self, theta):
        return np.sum(theta**2)

    def gradient(self, theta):
        return 2 * theta