from .l1 import L1
from .l2 import L2

class ElasticNet:
    """
        This class implements the Elastic Net Regularization
    """
    def __init__(self, l1_ratio=0.5, *args, **kwargs):
        self.l1_ratio = l1_ratio
        self.l1 = L1()
        self.l2 = L2()


    def regularize(self, theta):
        return self.l1_ratio * self.l1.regularize(theta) + (1 - self.l1_ratio) * self.l2.regularize(theta)

    def gradient(self, theta):
        return  self.l1_ratio * self.l1.gradient(theta) + (1 - self.l1_ratio) * self.l2.gradient(theta)