"""
problem 3
"""

import random_data_generator as rdg
import numpy as np
import sys
import matplotlib.pyplot as plt


def gaussian_probability(mean, var, x):
    """
    Calculate Gaussian probability.
    """
    stdev = np.sqrt(var)
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * np.power(stdev, 2))))
    return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent


def posterior_hyperparameters(n, x, y, a, mu0, precision0):
    """
    Compute the hyperparameters (mu and lambda) for Gaussian distribution of posterior (predicting w).
    :param n: number of basis
    :param x: x of data
    :param y: y of data
    :param a:
    :param mu0: mu of prior
    :param precision0: precision of prior
    :return: mu, lambda of posterior
    """
    A = np.zeros(n).reshape((1, n))
    for i in range(n):
        A[0, i] = x ** i
    precision = a * np.dot(np.transpose(A), A) + precision0
    mu = np.dot(np.linalg.inv(precision), a * y * np.transpose(A) + np.dot(precision0, mu0))
    return mu, precision


def predictive_posterior_hyperparameters(n, x, a, mu, precision):
    """
    Compute the hyperparameters of predictive posterior (predicting y),
    and generate y at the same time.
    :param n:
    :param x:
    :param a:
    :param mu:
    :param precision:
    :return:
    """
    phi = np.zeros((n, 1))
    for i in range(n):
        phi[i, 0] = x ** i

    p_mu = np.dot(np.transpose(mu), phi)
    p_var = 1 / a + np.dot( np.dot(np.transpose(phi), np.linalg.inv(precision)), phi)
    y = rdg.box_muller(p_mu, p_var)
    return p_mu, p_var, y


def p(x, y, n, a, mu, precision):
    """
    Return the probability of every point(x, y).
    """
    p_mu = np.zeros(x.shape)
    p_var = np.zeros(x.shape)
    p_y = np.zeros(len(x[0]))
    for i in range(len(x[0])):
        p_mu[:, i], p_var[:, i], p_y[i] = predictive_posterior_hyperparameters(n, x[0, i], a, mu, precision)

    return gaussian_probability(p_mu, p_var, y)


if __name__ == "__main__":
    n = 0
    a = 0.0
    if len(sys.argv) < 4:
        print("Usage:", sys.argv[0], "<precision_init_prior>", "<num_basis>", "<a>", "<w0>", "<w1> ...")
        print("Use default polynomial linear model: b=0.04 ,n=2, a=5, w0=0, w1=1")
        b = 0.04
        n = 2
        a = 5
        w = np.array([0, 1]).reshape((2, 1))
    else:
        b = float(sys.argv[1])
        n = int(sys.argv[2])
        a = float(sys.argv[3])
        W = []
        for i in range(4, len(sys.argv)):
            W.append(float(sys.argv[i]))
        w = np.array(W).reshape((n, 1))

    x = np.linspace(-10.0, 10.0, 20)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = rdg.linear_model_data_generator(n, a, w, x[i])

    mu = np.zeros((n, 1))
    precision = b * np.identity(n)
    for i in range(len(x)):
        index = int(np.random.random() * len(x))
        mu, precision = posterior_hyperparameters(n, x[index], y[index], a, mu, precision)

        if i % 2 == 0:
            print("Hyperparameters of posterior:")
            print("mu:")
            print(mu)
            print("precision:")
            print(precision)
            plt.scatter(x[index], y[index])
            x = np.linspace(-10, 10, 1000)
            y = np.linspace(-10, 10, 1000)
            X, Y = np.meshgrid(x, y)
            extent = [-10, 10, -10, 10]
            plt.imshow(p(X, Y, n, a, mu, precision), origin="lower", extent=extent)
            plt.colorbar()
            plt.show()
        print("--------------------------------------------")

    # print(mu)
    # print(precision)
    # pdf_x = np.linspace(-10.0, 10.0, 100)
    # predict_y = np.zeros(len(pdf_x))
    # for i in range(len(pdf_x)):
    #     t, tt, predict_y[i] = predictive_posterior_hyperparameters(n, pdf_x[i], a, mu, precision)
    # plt.scatter(pdf_x, predict_y, s=5)
    # plt.show()