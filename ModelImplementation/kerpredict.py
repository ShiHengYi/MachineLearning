import K
import numpy as np

# Input: numpy vector alpha of n rows, 1 column
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# numpy vector z of d rows, 1 column
# Output: label (+1 or -1)
def run(alpha,X,y,z):
    n = len(X)
    sum = 0
    for i in range (n):
        sum += 1.0* alpha[i]*y[i]*K.run(X[i], z)
    if sum > 0:
        return 1
    else:
        return -1
