import numpy as np
from sklearn.datasets import load_iris
from sklearn import datasets

def load_data():
    iris = datasets.load_iris()

    # take every fifth element as validation data required for training
    x_val = iris.data[0::5]
    y_val = iris.target[0::5]

    # take all data except for validation data as training data
    x_train = np.array([item for index, item in enumerate(iris.data) if index % 5 != 0])
    y_train = np.array([item for index, item in enumerate(iris.target) if index % 5 != 0])

    data = {'Xval': x_val, 'Xtrain': x_train, 'yval': y_val, 'ytrain' : y_train}
    return data


def predict(x, k):
    iris = datasets.load_iris()

    x_train = iris.data[:, :4]
    y_train = iris.target
    distance = euclid_distance(x, x_train)
    x_labels = sort_train_labels_knn(distance, y_train)
    p_matrix = p_y_x_knn(x_labels, k)
    result = choose_the_best(p_matrix)
    return result


def euclid_distance(x, x_train):
    """
    Return Euclidean distance for objects from set *x* from objects from set *x_train*.

    :param x: set of compared objects, dimensions N1xD
    :param x_train: set of objects function compares to, dimensions N2xD
    :return: matrix of distances between objects from "x" and "x_train" N1xN2
    """

    x = np.array(x)
    x_train = np.array(x_train)

    distance = np.zeros(shape=(x.shape[0], x_train.shape[0]))
    for i in range(distance.shape[0]):
        for j in range(distance.shape[1]):
            distance[i][j] = np.linalg.norm(x[i] - x_train[j])
    return distance


def p_y_x_knn(y, k):
    """
    Calculate probability distribution p(y|x) for each class for objects
    from test dataset using KNN trained on training data.

    :param y: matrix of sorted labels for training data, dimensions N1xN2
    :param k: count of nearest neighbours for KNN
    :return: matrix of probabilities p(y|x) for objects from "x", dimensions N1xM
    """
    NUMBER_OF_CLASSES = 3
    A = np.zeros(shape=(y.shape[0], NUMBER_OF_CLASSES))
    y = y[:, :k]

    for i in range(y.shape[0]):
        for j in range(NUMBER_OF_CLASSES):
            A[i][j] = np.count_nonzero(y[i] == j)
    A /= k
    return A


def choose_the_best(p_matrix):
    """
    :param p_matrix: matrix containing probabilities of affiliation to
                    each class for every processed object
   
    :return: matrix of recognized classes for input data
    """

    result = np.zeros(shape=(p_matrix.shape[0], 1))
    for i in range(result.shape[0]):
        result[i] = np.argmax(p_matrix[i])
    return result.astype(np.uint)


def classification_error(p_y_x, y_true):
    """
    Calculate classification error.

    :param p_y_x: matrix of probabilities - each row contains
                distribution of p(y|x), dimension NxM
    :param y_true: set of valid class labels, dimensions 1xN
    :return: classification error
    """
    error = 0
    for i in range(len(y_true)):
        p = 0
        idx = 0
        for j in range(p_y_x.shape[1]):
            if p_y_x[i][j] >= p:
                p = p_y_x[i][j]
                idx = j
        if idx != y_true[i]:
            error += 1

    return error / len(y_true)


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Calculate error for various *k* values. Choose KNN model by
    determining the best *k* value, which is the value with the least error.

    :param X_val: validation dataset N1xD
    :param X_train: training dataset N2xD
    :param y_val: class labels for validation data 1xN1
    :param y_train: class labels for training data 1xN2
    :param k_values: k values to be checked
    :return: tuple (least_error, best_k, errors), is the lowest error reached
    , "best_k" is the lowest "k" for which error was lowest,
    and "errors" - list of error values for each "k" from "k_values"
    """
    dist = euclid_distance(X_val, X_train)
    sort_labels = sort_train_labels_knn(dist, y_train)
    errors = []
    k = len(k_values)
    for i in range(k):
        probability = p_y_x_knn(sort_labels, k_values[i])
        errors.append(classification_error(probability, y_val))

    least_error = min(errors)
    index = errors.index(least_error)
    best_k = k_values[index]

    return least_error, best_k, errors

def sort_train_labels_knn(dist, y):
    """
    Sort class labels *y* for training data by distances
    included in matrix *dist*.

    :param dist: matrix of distances between objects 
        from "X" and "X_train", dimensions N1xN2
    :param y: vector of class labels (targets) of length N2
    :return: matrix of class labels sorted by probability 
        values of corresponding row of dist matrix N1xN2

    Using mergesort algorithm for sorting.
    """
    
    indices = dist.argsort(kind='mergesort')
    return y[indices]


def run_training():
    data = load_data()

    # KNN model selection
    k_values = range(1, 101, 2)
    print('\n---------------- Finding best neighbours number ----------------')
    print('------------------- Values k: 1, 3, ..., 100 ----------------------')

    error_least, best_k, errors = model_selection_knn(data['Xval'],
                                                     data['Xtrain'],
                                                     data['yval'],
                                                     data['ytrain'],
                                                     k_values)
    print('Errors for all *k* values: ')
    for i in range(len(k_values)):
        print('k = {k}, error = {err}'.format(k=k_values[i], err=errors[i]))
    print('Best k: {num1}\nLeast error: {num2:.4f}'.format(num1=best_k, num2=error_least))



if __name__ == '__main__':
        run_training()