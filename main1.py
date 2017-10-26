import random_data_generator as rdg
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sys


num_data = 5000


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:", sys.argv[0], "<mean>", "variance")
        print("Use default normal distribution: mean=0, var=1")
        mean = 0
        var = 1
    else:
        mean = float(sys.argv[1])
        var = float(sys.argv[2])

    x = np.zeros(num_data)
    for i in range(num_data):
        x[i] = rdg.box_muller(mean, var)

    x_min = mean - 4 * var
    x_max = mean + 4 * var

    # histogram
    mu, sigma = mean, var

    # the histogram of the data
    n, bins, patches = plt.hist(x, 100, normed=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line (red line)
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('Data')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ data:}\ \mu=%.2f,\ \sigma=%.2f$' % (mu, sigma))
    plt.axis([x_min, x_max, 0, 1])
    plt.grid(True)

    plt.show()
