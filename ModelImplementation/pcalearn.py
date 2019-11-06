import numpy as np
from numpy import cov
from numpy.linalg import eig
import numpy.linalg as la
# Input: number of features F
# numpy matrix X, with n rows (samples), d columns (features)
# Output: numpy vector mu, with d rows, 1 column
# numpy matrix Z, with d rows, F columns
def run(F,X):
    X = np.copy(X)
    d = len(X[0])
    n = len(X)
    sums = np.sum(X, axis=0)
    mu = np.zeros(d)

    for i in range(d):
        mu[i] += 1.0 / n * sums[i]

    for t in range(n):
        for i in range(d):
            X[t][i] -= mu[i]

    U, s, Vt = la.svd(X, False)
    g = s[:F]

    for i in range(F):
        if g[i] > 0:
            g[i] = 1.0/g[i]
    W = Vt[:F]
    Z = np.dot(W.T, np.diag(g))

    return (mu.reshape(-1,1), Z)

