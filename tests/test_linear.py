import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from pymlscratch.linear.softmax_reg import SoftmaxRegression
from pymlscratch.linear.linear_reg import LinearRegression
from pymlscratch.linear.linear_reg_gd import LinearRegressionGD
from pymlscratch.linear.linear_reg_sgd import LinearRegressionSGD
from pymlscratch.linear.ridge_reg import RidgeRegression
from pymlscratch.linear.lasso_reg import LassoRegression
from pymlscratch.linear.elastic_net_reg import ElasticNetRegression
from pymlscratch.linear.logistic_reg import LogisticRegression

"""
    This file contains unit tests for the linear models.
"""

def get_data():
    np.random.seed(42)  # to make this code example reproducible
    m = 100  # number of instances
    X = 2 * np.random.rand(m, 1)  # column vector
    y = 4 + 3 * X + np.random.randn(m, 1)  # column vector
    return X, y

def get_data_2():
    np.random.seed(42)
    m = 20
    X = 3 * np.random.rand(m, 1)
    y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
    return X, y

def test_linear_reg():
    X, y = get_data()
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    pred = lin_reg.predict(np.array([[0], [2]]))
    target = np.array([[4], [10]])
    np.testing.assert_allclose(pred, target, atol=0.5, err_msg='Expected: {}, Actual: {}'.format(target, pred),
                               verbose=True)


def test_linear_reg_gd():
    X, y = get_data()
    lin_reg = LinearRegressionGD(epochs=1000, eta=0.1)
    lin_reg.fit(X, y)
    pred = lin_reg.predict(np.array([[0], [2]]))
    target = np.array([[4], [10]])
    np.testing.assert_allclose(pred, target, atol=0.5, err_msg='Expected: {}, Actual: {}'.format(target, pred),
                               verbose=True)


def test_linear_reg_sgd():
    X, y = get_data()
    lin_reg = LinearRegressionSGD(epochs=1000, eta=0.01)
    lin_reg.fit(X, y)
    pred = lin_reg.predict(np.array([[0], [2]]))
    target = np.array([[4], [10]])
    np.testing.assert_allclose(pred, target, atol=0.5, err_msg='Expected: {}, Actual: {}'.format(target, pred),
                               verbose=True)

def test_linear_reg_gd_mini_batch():
    X, y = get_data()
    lin_reg = LinearRegressionGD(epochs=1000, eta=0.01, batch=16)
    lin_reg.fit(X, y)
    pred = lin_reg.predict(np.array([[0], [2]]))
    target = np.array([[4], [10]])
    np.testing.assert_allclose(pred, target, atol=0.5, err_msg='Expected: {}, Actual: {}'.format(target, pred),
                               verbose=True)

def test_ridge_reg():
    X, y = get_data_2()
    ridge_reg = RidgeRegression(alpha=0.1)
    ridge_reg.fit(X, y)
    pred = ridge_reg.predict([[1.5]])
    target = np.array([[1.55071465]])
    np.testing.assert_allclose(pred, target, atol=0.5, err_msg='Expected: {}, Actual: {}'.format(target, pred),
                               verbose=True)

def test_lasso_reg():
    X, y = get_data_2()
    lasso_reg = LassoRegression(epochs=1000, alpha=0.01)
    lasso_reg.fit(X, y)
    pred = lasso_reg.predict([[1.5]])
    target = np.array([[1.53788174]])
    np.testing.assert_allclose(pred, target, atol=0.5, err_msg='Expected: {}, Actual: {}'.format(target, pred),
                               verbose=True)


def test_elastic_net_reg():
    X, y = get_data_2()
    elastic_net_reg = ElasticNetRegression(epochs=10000, alpha=0.01, l1_ratio=0.5)
    elastic_net_reg.fit(X, y)
    pred = elastic_net_reg.predict([[1.5]])
    target = np.array([[1.54333232]])
    np.testing.assert_allclose(pred, target, atol=0.5, err_msg='Expected: {}, Actual: {}'.format(target, pred),
                               verbose=True)

def test_logistic_reg():
    iris = load_iris(as_frame=True)
    X = iris.data[["petal width (cm)"]].values
    y = iris.target_names[iris.target] == 'virginica'
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    log_reg = LogisticRegression(epochs=100000, eta=0.1, C=100)
    log_reg.fit(X_train, y_train)
    pred = log_reg.predict(np.array([[1.7], [1.5]]))
    target = np.array([True, False])
    np.testing.assert_equal(pred, target, err_msg='Expected: {}, Actual: {}'.format(target, pred))

def test_softmax_reg():
    iris = load_iris(as_frame=True)
    X = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y = iris['target'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    softmax_reg = SoftmaxRegression(epochs=100000, eta=0.1, alpha=0.1)
    softmax_reg.fit(X_train, y_train)
    pred = softmax_reg.predict([[5, 2]])
    target = np.array([2])
    np.testing.assert_equal(pred, target, err_msg='Expected: {}, Actual: {}'.format(target, pred))