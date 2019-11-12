import numpy as np
import numpy.linalg as la
# Input: number of iterations L
# number of clusters k
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# Output: numpy vector a of n rows, 1 column
# a[i] is the cluster assignment for the i-th sample (an integer from 0 to k-1)
# number of iterations that were actually executed (iter+1)
def run(L,k,X):
    n = len(X)
    d = len(X[0])
    a = np.zeros(n)
    q = np.random.choice(n,k,replace=False)
    r = np.zeros((k,d))
    for j in range(k):
        r[j] = X[q[j]]
    for iter in range(L):
        at_least_one_change = False
        for t in range(n):
            c = 0
            b = la.norm(X[t] - r[0])
            for j in range(1, k):
                if la.norm(X[t] - r[j]) < b:
                    c = j
                    b = la.norm(X[t] - r[j])
            if a[t] != c:
                a[t] = c
                at_least_one_change = True

        if not at_least_one_change:
            break
        for j in range(k):
            s = np.zeros(d)
            m = 0
            for t in range(n):
                if a[t] == j:
                    s += X[t]
                    m += 1
            r[j] = 1.0*s/m
    return a.reshape(-1,1), iter+1