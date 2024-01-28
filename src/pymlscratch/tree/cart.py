import numpy as np
from scipy import stats

class CartNode:
    def __init__(self, feature, feature_threshold):
        self.feature = feature
        self.feature_threshold = feature_threshold
        self.left: None = None
        self.right: None = None
        self.is_leaf: bool = False
        self.target_value = None
        self.impurity_score = None

class Cart:
    def __init__(self, impurity, min_sample_size, max_depth, calculate_target_value_fn, *args, **kwargs):
        self.root = None
        self.impurity = impurity
        self.min_sample_size = min_sample_size
        self.max_depth = max_depth
        self.calculate_target_value_fn = calculate_target_value_fn

    def should_continue(self, X, y, index_mask, level):
        return not (np.sum(index_mask) <= self.min_sample_size or level == self.max_depth or
         self.impurity.score(X[index_mask], y[index_mask]) == 0)
    def grow_tree(self, X, y) -> None:
        def _create(X, y, index_mask, level, features):
            if not self.should_continue(X, y, index_mask, level):
                node = CartNode(None, None)
                node.target_value = self.calculate_target_value_fn(y[index_mask])
                node.impurity_score = self.impurity.score(X[index_mask], y[index_mask])
                node.is_leaf = True
                return node
            best_feature = None
            best_feature_threshold = None
            best_impurity_score = np.inf
            for feature, flag in enumerate(features):
                if not flag:
                    continue
                unique_values = np.linspace(np.min(X[:, feature]), np.max(X[:, feature]))
                for unique_value in unique_values:
                    impurity_score = 0
                    mask = np.where(X[:, feature] < unique_value, True, False)
                    left_mask = index_mask & mask
                    right_mask = index_mask & ~mask

                    if np.sum(left_mask) < self.min_sample_size or np.sum(right_mask) < self.min_sample_size:
                        continue

                    X_left = X[left_mask]
                    y_left = y[left_mask]
                    impurity_score += (X_left.shape[0] / X.shape[0]) * self.impurity.score(X_left, y_left)

                    X_right = X[right_mask]
                    y_right = y[right_mask]
                    impurity_score += (X_right.shape[0] / X.shape[0]) * self.impurity.score(X_right, y_right)

                    if impurity_score < best_impurity_score:
                        best_impurity_score = impurity_score
                        best_feature = feature
                        best_feature_threshold = unique_value
            if best_feature is None:
                return None
            mask = np.where(X[:, best_feature] <= best_feature_threshold, True, False)
            left_mask = index_mask & mask
            right_mask = index_mask & ~mask

            features[best_feature] = False

            node = CartNode(best_feature, best_feature_threshold)
            node.left = _create(X, y, left_mask, level+1, features)
            node.right = _create(X, y, right_mask, level+1, features)
            if node.left is None and node.right is None:
                node.is_leaf = True
                node.majority_class = stats.mode(y[index_mask])

            return node
        features = np.ones(X.shape[1], dtype=bool)
        index_mask = np.ones(X.shape[0], dtype=bool)
        self.root = _create(X, y,  index_mask,0, features)

    def traverse_tree(self, X):
        predictions = np.zeros(X.shape[0])
        def _traverse(node: CartNode, X,  index_mask):
            if node.is_leaf:
                predictions[index_mask] = node.target_value
                return

            mask = np.where(X[:, node.feature] <= node.feature_threshold, True, False)
            left_mask = index_mask & mask
            right_mask = index_mask & ~mask

            _traverse(node.left, X, left_mask)
            _traverse(node.right, X, right_mask)
        _traverse(self.root, X, np.ones(X.shape[0], dtype=bool))
        return predictions