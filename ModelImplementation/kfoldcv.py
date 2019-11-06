import numpy as np
from probcpredict import run as probcpredict
from probclearn import run as probclearn
# Input: number of folds k
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of k rows, 1 column
def run(k,X,y):
    n = len(X)
    z = np.zeros(k)
    for i in range(k):
        T = list(range(int(1.0*n*i/k), int(1.0*n*(i + 1)/k) ) )
        S = list(set(list(range(0, n))) - set(T))
        X_train = X[S]
        Y_train = y[S]
        q,up,un,sp,sn = probclearn(X_train, Y_train)

        for t in T:
            if y[t] != probcpredict(q,up,un,sp,sn, X[t].reshape(-1,1)):
                z[i] = z[i] + 1
        z[i] = z[i] / len(T)
    return z.reshape(-1,1)