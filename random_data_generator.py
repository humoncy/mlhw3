import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sys


generate = 0
z1 = 0


def box_muller(mean, var):
    """
    Box-Muller transform - univariate Gaussian data generator
    """
    global generate
    global z1

    generate = ~generate

    if not generate:
        return mean + z1 * var

    u, v = 0.0, 0.0
    while u < sys.float_info.epsilon:
        u = np.random.random()
        v = np.random.random()

    z0 = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)
    z1 = np.sqrt(-2 * np.log(u)) * np.sin(2 * np.pi * v)
    return mean + z0 * var


# def polybasis_linearmodel_data_generator(n, a, w):
