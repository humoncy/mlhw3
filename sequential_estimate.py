"""
problem 2
"""
import numpy as np
from random_data_generator import box_muller
import sys


def online_mean(xn, old_mean, n):
    delta = xn - old_mean
    new_mean = old_mean + (delta / n)
    return new_mean


def sequential_estimation(mu, sigma):
    x = []
    mean = M2 = 0.0
    i = 0
    error = sys.maxsize
    while error > 0.0001:
        n = i + 1
        x.append(box_muller(mu, sigma))
        print("data point: ", x[i])
        delta = x[i] - mean
        mean += delta / n
        # mean = np.mean(x)
        print("mean: ", mean)

        delta2 = x[i] - mean
        M2 += delta * delta2

        if n < 2:
            var = np.var(x)
        else:
            var = M2 / (n - 1)
        # var = np.var(x)
        print("variance: ", var)

        print("----------------------------------")
        error = np.abs(mean - mu)
        i += 1
    print("number of iterations: ", i)


sequential_estimation(0, 1)
