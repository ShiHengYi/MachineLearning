import numpy as np
from numpy import cov
from numpy.linalg import eig
import numpy.linalg as la

# Input: number of features F
# numpy matrix X, with n rows (samples), d columns (features)
# numpy vector mu, with d rows, 1 column
# numpy matrix Z, with d rows, F columns
# Output: numpy matrix P, with n rows, F columns
def run(X,mu,Z):
    X = np.copy(X)
    n = len(X)
    d = len(X[0])

    for t in range(n):
        for i in range(d):
            X[t][i] = X[t][i] - mu[i]
    P = np.dot(X,Z)
    return P