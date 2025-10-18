import numpy as np


def mean_squared_error(y_true, y_predicted):
    # Calculating the loss or cost
    cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
    return cost


def univariate_gradient_descent(x, y, iterations, learning_rate):
    m = 1
    b = 1
    losses = []
    iterations = iterations
    learning_rate = learning_rate

    for i in range(0, iterations):
        m_gradient = 0
        b_gradient = 0
        for s in range(0, len(x)):
            error = y[s] - (m * x[s] + b)
            m_gradient += -2 * x[s] * error
            b_gradient += -2 * error
        m = m - learning_rate * m_gradient / len(x)
        b = b - learning_rate * b_gradient / len(x)
        y_pred = m * x + b
        loss = mean_squared_error(y, y_pred)
        losses.append(loss)

    return m, b,losses



def multivariate_gradient_descent(X, y, iterations, learning_rate):
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    b = 0

    for i in range(iterations):
        weight_gradients = np.zeros(n_features)
        b_gradient = 0

        for s in range(len(X)):
            error = y[s] - (np.dot(X[s], weights) + b)
            weight_gradients += -2 * X[s] * error
            b_gradient += -2 * error

        weights = weights - learning_rate * weight_gradients / len(X)
        b = b - learning_rate * b_gradient / len(X)

    return weights, b
