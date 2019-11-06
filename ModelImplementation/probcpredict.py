from numpy import linalg as LA
import math
import numpy as np
# Input: scalar q
# numpy vector mu_positive of d rows, 1 column
# numpy vector mu_negative of d rows, 1 column
# scalar sigma2_positive
# scalar sigma2_negative
# numpy vector z of d rows, 1 column
# Output: label (+1 or -1)
def run(q,mu_positive,mu_negative,sigma2_positive,sigma2_negative,z):
    p1 = math.log(q/(1-q))
    p2 = len(mu_positive)/2 *math.log(sigma2_positive/sigma2_negative)
    p3 = 1/(2*sigma2_positive)*LA.norm(z-mu_positive)
    p4 = 1/(2*sigma2_negative)*LA.norm(z-mu_negative)
    total = p1 - p2 - p3 + p4
    if total > 0:
        return 1
    else:
        return -1