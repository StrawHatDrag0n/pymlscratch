import numpy as np

class Softmax:
    """
        This class implements the softmax function.
    """
    @classmethod
    def forward(cls, X, eps=1e-8):
        X -= np.max(X, axis=1, keepdims=True)
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
    @classmethod
    def gradient(cls, y, scores):
        return scores - y
