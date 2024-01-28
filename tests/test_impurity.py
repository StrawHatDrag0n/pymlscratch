import numpy as np
from pymlscratch.impurity.gini import GiniImpurity
from sklearn.datasets import load_iris

def test_gini_impurity():
    iris = load_iris(as_frame=True)
    X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y_iris = iris.target
    mask = np.where(X_iris[:, 0] <= 2.45, True, False)

    X_iris_mask = X_iris[mask]
    y_iris_mask = y_iris[mask]
    target = 0.0
    np.testing.assert_allclose(target, GiniImpurity.impurity(X_iris_mask, y_iris_mask))

    X_iris_no_mask = X_iris[~mask]
    y_iris_no_mask = y_iris[~mask]
    target = 0.5
    np.testing.assert_allclose(target, GiniImpurity.impurity(X_iris_no_mask, y_iris_no_mask))