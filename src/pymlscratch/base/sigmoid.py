import numpy as np


class Sigmoid:
    @classmethod
    def forward(cls, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.forward(x) * (1 - self.forward(x))
    