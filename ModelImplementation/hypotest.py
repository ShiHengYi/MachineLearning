from scipy.stats import t
import numpy as np
import math
# Input: numpy vector a of scalar values, with k rows, 1 column
# numpy vector b of scalar values, with k rows, 1 column
# scalar alpha
# Output: reject (0 or 1)
def run(a,b,alpha):
    u1 = np.mean(a)
    u2 = np.mean(b)
    s1 = np.var(a)
    s2 = np.var(b)
    x = (u1 - u2)*math.sqrt(len(a))/ math.sqrt(s1 + s2)
    v = math.ceil( (s1 + s2)**2 * (len(a)-1) /(s1**2 + s2**2) )

    if x > t.ppf(1-alpha, v):
        reject = 1
    else:
        reject = 0
    
    return reject