import numpy as np
from numpy import linalg as LA
import math

# Input: numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# numpy vector z of d rows, 1 column
# Output: label (+1 or -1)
def run(X,y,z):
    c = 0
    z = z.reshape(1,len(z))
    b = LA.norm(z-X[0])
    for t in range(1, len(y)):
        if LA.norm(z-X[t]) < b:
            c = t
            b = LA.norm(z-X[t])
    label = y[c]
    if label == 1:
        return 1
    else:
        return -1