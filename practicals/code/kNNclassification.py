import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(a-b) #Use norm to get the size of the vector

def get_neighbours(X_train, X_test, k):
    distances = np.zeros(len(X_train))
    for i in range(len(X_train)):
        distances[i] = euclidean_distance(X_test, X_train[i])
    index = np.argsort(distances)
    distances = np.sort(distances)
    return index[:k], distances[:k]

def kNN_classification_pred(X_train, X_test, k):
    y_pred = np.zeros(len(X_test))
    for i, x_test in enumerate(X_test):
        index, dist = get_neighbours(X_train, x_test, k)
        if np.sum(y_train[index,0]) > k/2:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred[:, np.newaxis]

def get_kNN_accuracy(X_test, X_train, y_test, k_lin):
    accuracy = np.zeros(len(k_lin))
    for i, k in enumerate(k_lin):
        y_pred = get_predictions(X_train, X_test, k)
        accuracy[i] = sum(y_pred[:,0] == y_test[:,0]) / len(y_test[:,0])
    return accuracy