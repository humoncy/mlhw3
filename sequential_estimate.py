import numpy as np
from random_data_generator import box_muller


def sequential_estimation(mu, sigma):
    x = []
    for i in range(10000):
        x.append(box_muller(mu, sigma))
        mean = np.mean(x)
        var = np.var(x)
        print("data point: ", x[i])
        print("mean: ", mean)
        print("variance: ", var)

        if np.abs(mean - mu) < 0.0001:
            print("number of iterations: ", i)
            break
        print("----------------------------------")


sequential_estimation(0, 1)
