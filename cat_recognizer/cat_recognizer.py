import numpy as np
import copy
from lr_utils import load_dataset
from util import basic_math as bm
import testing


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above
    :param w: weights, a numpy array of size (num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of size (num_px * num_px * 3, number of examples)
    :param Y: true "label" vector of size (1, number of examples)
    :returns:
    cost: negative log-likelihood cost for logistic regression
    dw: gradient of the loss with respect to w, thus same shape as w
    db: gradient of the loss with respect to b, thus same shape as b
    """

    m = X.shape[1]
    A = bm.sigmoid(np.dot(w.T, X) + b)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A), keepdims=True)

    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    :param w: weights, a numpy array of size (num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of shape (num_px * num_px * 3, number of examples)
    :param Y: true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    :param num_iterations: number of iterations of the optimization loop
    :param learning_rate: learning rate of the gradient descent update rule
    :param print_cost: True to print the loss every 100 steps

    :returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]
        w -= learning_rate * dw
        b -= learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    :param w: weights, a numpy array of size (num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of size (num_px * num_px * 3, number of examples)

    :return: Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = bm.sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    :param X_train: training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    :param Y_train: training labels represented by a numpy array (vector) of shape (1, m_train)
    :param X_test: test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    :param Y_test: test labels represented by a numpy array (vector) of shape (1, m_test)
    :param num_iterations: hyperparameter representing the number of iterations to optimize the parameters
    :param learning_rate: hyperparameter representing the learning rate used in the update rule of optimize()
    :param print_cost: Set to True to print the cost every 100 iterations

    :returns: d -- dictionary containing information about the model.
    """
    w, b = np.zeros((X_train.shape[0], 1)), 0.0
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
    w, b = params['w'], params['b']
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


def main():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    num_px = train_set_x_orig[0].shape[0]

    train_set_x_flatten = bm.flatten(train_set_x_orig)
    test_set_x_flatten = bm.flatten(test_set_x_orig)
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000,
                                      learning_rate=0.005, print_cost=True)

    testing.plot_learning_curve(logistic_regression_model)
    testing.test_image('IMG_2652.JPEG', logistic_regression_model, classes, num_px)


if __name__ == '__main__':
    main()
