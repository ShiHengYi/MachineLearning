import numpy as np

# Input: maximum number of iterations L
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector theta of d rows, 1 column
# number of iterations that were actually executed (iter+1)
def run(L, X, y):
    
    theta = np.zeros(len(X[0]))
    iter = 0
    for i in range(L):
    	all_points_classified_correctly = 1
        for j in range(len(X)):
            if (np.dot(X[j], theta) * y[j]) <= 0:
                theta = theta + y[j] * X[j]
                all_points_classified_correctly = -1
        if all_points_classified_correctly == 1:
        	break
        else:
        	iter = i      	    	                

    theta = theta.reshape((-1, 1))
    return theta, iter + 1