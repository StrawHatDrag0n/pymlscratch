import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris, make_blobs
from pymlscratch.svm.svc import SVC


def get_blob_data():
    X, y = make_blobs(
        n_samples=250, n_features=2, centers=2, cluster_std=1.05, random_state=1
    )
    return X, y

def get_iris_data():
    iris = load_iris(as_frame=True)
    X = iris.data[["petal length (cm)", "petal width (cm)"]].values
    X = StandardScaler().fit_transform(X)
    y = (iris.target == 2)  # Iris virginica
    y = y.to_numpy()
    return X, y


def test_svc():
    X, y = get_blob_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    svc = SVC(epoch=1000, eta=0.001, C=100)
    svc.fit(X_train, y_train)
    pred = svc.predict(X_test)
    print("Test Accuracy: ", accuracy_score(y_test, pred))
