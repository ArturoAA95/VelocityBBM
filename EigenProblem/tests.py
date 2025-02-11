import numpy as np
from EigenProblem import PerOp_FD
from BranchingRates import g_1

n = 3
e = [0,1]
lamb = 1


A = PerOp_FD(n, e, lamb, g_1)

print(np.diag(A))