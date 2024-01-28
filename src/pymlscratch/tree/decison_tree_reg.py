from ..impurity.impurity_factory import ImpurityFactory
from ..tree.tree_growing_algo_factory import TreeGrowingAlgoFactory
from ..tree.utils import calculate_mean

class DecisionTreeRegressor:
    def __init__(self, impurity='MAE', min_samples_leaf=1, max_depth=None, splitter='best'):
        self.impurity = ImpurityFactory.create(impurity)
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.tree_growing_algo = TreeGrowingAlgoFactory.create(algo='CART',
                                                               impurity=self.impurity,
                                                               min_samples_leaf=self.min_samples_leaf,
                                                               max_depth=self.max_depth,
                                                               calculate_target_value=calculate_mean)

    def fit(self, X, y):
        self.tree_growing_algo.grow_tree(X, y)
        return self

    def predict(self, X):
        return self.tree_growing_algo.traverse_tree(X)