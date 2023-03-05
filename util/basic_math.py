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



