import cvxopt as co
import numpy as np

# Input: numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector theta of d rows, 1 column
def run(X,y):
    
    H = np.identity(len(X[0]))
    f = np.zeros(len(X[0]))
    A = np.zeros((len(y), len(X[0])))
    b = -1 * np.ones(len(y))

    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = 0 - np.dot(y[i], X[i][j])

    theta = np.array(co.solvers.qp(co.matrix(H, tc='d'), co.matrix(f, tc='d'), co.matrix(A, tc='d'), co.matrix(b, tc='d'))['x'])
    return theta