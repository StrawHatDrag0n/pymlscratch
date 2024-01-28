import numpy as np
from pymlscratch.tree.decision_tree_clf import DecisionTreeClassifier
from pymlscratch.tree.decison_tree_reg import DecisionTreeRegressor
from sklearn.datasets import load_iris


def test_decision_tree_classifier():
    X = np.array([[0, 0], [1, 1]])
    Y = np.array([0, 1])
    clf = DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    print(clf.predict(np.array([[2., 2.]])))


def test_decision_tree_regressor():
    rng = np.random.RandomState(1)
    X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += 0.5 - rng.rand(20, 2)
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_1.fit(X, y)