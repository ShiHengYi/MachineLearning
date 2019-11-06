import K
import numpy as np

# Input: number of iterations L
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector alpha of n rows, 1 column
# number of iterations that were actually executed (iter+1)
def run(L,X,y):
    n = len(X)
    alpha = np.zeros(n)

    for iter in range(L):
        all_point_classified_correctly = 1
        for t in range(n):
            sum = 0
            for i in range (n):
                sum += 1.0* alpha[i]*y[i]*K.run(X[i],X[t])
            if 1.0*y[t]*sum <= 0:
                alpha[t] += 1
                all_point_classified_correctly = 0
        if all_point_classified_correctly:
            break

    return alpha, iter+1

