import numpy as np
from numpy import linalg as LA
import math
# Input: numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: scalar q
# numpy vector mu_positive of d rows, 1 column
# numpy vector mu_negative of d rows, 1 column
# scalar sigma2_positive
# scalar sigma2_negative
def run(X,y):
    kp = 0
    kn = 0
    mu_positive = np.zeros(len(X[0]))
    mu_negative = np.zeros(len(X[0]))
    for t in range(len(y)):
        if y[t] == 1:
            kp = kp + 1
            mu_positive = np.add(mu_positive, X[t])
        else:
            kn = kn + 1
            mu_negative = np.add(mu_negative, X[t])

    q = kp / len(y)
    mu_positive = (1/kp)*mu_positive
    mu_negative = (1/kn)*mu_negative

    sigma2_positive = 0
    sigma2_negative = 0

    for t in range(len(y)):
        if y[t] == 1:
            sigma2_positive += math.pow(LA.norm(X[t] - mu_positive),2)
        else:
            sigma2_negative += math.pow(LA.norm(X[t] - mu_negative),2)


    sigma2_positive *= (1 / (len(X[0]) * kp))
    sigma2_negative *= (1 / (len(X[0]) * kn))

    mu_positive = mu_positive.reshape((-1, 1))
    mu_negative = mu_negative.reshape((-1, 1))
    return q, mu_positive, mu_negative, sigma2_positive, sigma2_negative