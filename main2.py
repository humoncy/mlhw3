"""
problem 1.(b)
"""

import random_data_generator as rdg
import numpy as np
import sys


if __name__ == "__main__":
    n = 0
    a = 0.0
    if len(sys.argv) < 3:
        print("Usage:", sys.argv[0], "<num_basis>", "<a>", "<w0>", "<w1> ...")
        print("Use default polynomial linear model: n=2, a=5, w0=0, w1=1")
        n = 2
        a = 5
        w = np.array([0, 1]).reshape((2, 1))
    else:
        n = int(sys.argv[1])
        a = float(sys.argv[2])
        W = []
        for i in range(3, len(sys.argv)):
            W.append(float(sys.argv[i]))
        w = np.array(W).reshape((n, 1))

    rdg.polybasis_linearmodel_data_generator(n, a, w)
