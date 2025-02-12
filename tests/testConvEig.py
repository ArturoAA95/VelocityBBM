import numpy as np
from EigenProblem import PerOp_FD
from EigenProblem import PerOp_FD_Sparse
from BranchingRates import g_1
import matplotlib.pyplot as plt
from scipy.sparse import isspmatrix_dia
from scipy.sparse import linalg
from scipy import integrate


e = [0,1]
lamb = 0
m = 25
m_s = 10
Eig = np.empty(m)

#principal eigenvalue when lambda=0 is r
r = integrate.nquad(g_1 , [[0,1], [0,1]])

#eigenvalue approx for smaller mesh-size
for i in range(m):
    n = (i+1)*m_s
    B = PerOp_FD_Sparse(n, e, lamb, g_1)
    b = linalg.eigs(B, 1 , sigma=r[0])
    Eig[i] = b[0][0]


#print(r[0])
#print(len(x))

x = r[0]*np.ones(m)
print(np.abs(r[0]-Eig[m-1] ))
plt.plot(Eig)
plt.plot(x)
plt.show()




