import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid function
    :param x: A scalar or numpy array
    :return: the computed sigmoid on the input
    """
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    """
    Compute the derivative of the sigmoid function with respect to its input x
    :param x: A scalar or numpy array
    :returns: the computed gradient
    """
    s = d_sigmoid(x)
    ds = s * (1 - s)
    return ds


def normalize_rows(mat):
    """
    Implement a function that normalizes each row of the matrix mat (to have unit length)
    :param mat: A numpy matrix of shape (n, m)
    :returns: The normalized (by row) numpy matrix
    """
    mat_norm = np.linalg.norm(mat, ord_mat=2, axis=1, keepdims=True)
    mat_normalized = mat / mat_norm
    return mat_normalized


def softmax(mat):
    """
    Calculates the softmax for each row of the input mat
    :param mat: A numpy matrix of shape (m,n)
    :returns: A numpy matrix equal to the softmax of mat, of shape (m,n)
    """
    mat_exp = np.exp(mat)
    mat_sum = np.sum(mat_exp, axis=1, keepdims=True)
    s = mat_exp/mat_sum
    return s


def vectorize(x: np.ndarray):
    """
    Vectorize the N dimensional object into 1 dimensional row vector
    :param x: the N dimensional numpy array
    :return: the vectorization the object
    """
    size = 1
    for i in x.shape:
        size *= i
    return x.reshape((1, size))


def L1(y, y_hat):
    """
    Calculates the loss between two given outputs as the sum of the differences
    :param y: the expected output
    :param y_hat: real output
    :return: the loss between the real and expected outputs
    """
    return np.sum(np.abs(y - y_hat))


def L2(y, y_hat):
    """
    Calculates the loss between two given outputs as the sum of the squared differences
    :param y: the expected output
    :param y_hat: real output
    :return: the loss between the real and expected outputs
    """
    return np.sum((y - y_hat) ** 2)


def flatten(nd_array: np.ndarray):
    return nd_array.reshape(nd_array.shape[0], -1).T
