import numpy as np


class GiniImpurity:
    @classmethod
    def score(cls, X, y):
        unique_ys = np.unique(y)
        impurity_score = 0
        for unique_y in unique_ys:
            impurity_score += (np.sum((np.where(y == unique_y, 1, 0)) / y.shape[0])) ** 2
        return 1 - impurity_score