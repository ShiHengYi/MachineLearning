import numpy as np
from probcpredict import run as probcpredict
from probclearn import run as probclearn
# Input: number of bootstraps B
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of B rows, 1 column
def run(B,X,y):
    n = len(X)
    z = np.zeros(B)
    for i in range(B):
        u = np.zeros(n)
        S = set()
        for j in range(n):
            k = int(np.random.randint(0,n,1))
            u[j] = k
            S = S.union({k})
        T = list(set(list(range(n))) - set(S))
        u = u.astype(int)
        ul = u.tolist()
        X_train = X[ul]
        Y_train = y[ul]
        q, up, un, sp, sn = probclearn(X_train, Y_train)
        for t in T:
            if y[t] != probcpredict(q, up, un, sp, sn, X[t].reshape(-1, 1)):
                z[i] = z[i] + 1
        z[i] = z[i] / len(T)
    return z.reshape(-1,1)