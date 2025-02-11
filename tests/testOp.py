import numpy as np
from EigenProblem import PerOp_FD
from EigenProblem import PerOp_FD_Sparse
from BranchingRates import g_1
import matplotlib.pyplot as plt
from scipy.sparse import isspmatrix_dia
from scipy.sparse import linalg

n = 50
e = [0,1]
lamb = 1


#A = PerOp_FD(n, e, lamb, g_1)
B = PerOp_FD_Sparse(n, e, lamb, g_1)
b = linalg.eigs(B, 1 , which='LR')

#print(isspmatrix_dia(B))
#C = np.matrix( B.toarray())
#print(np.linalg.norm(B-C))

#print(b[0])
#C = np.matrix(B.toarray())

#print(np.diag(A))

#m = np.diag(A,k=n**2-1)
#m = np.diag(A,k=n**2-50)
#m = np.diag(A,k=-n**2+50)
#m = np.diag(A,k=-n**2+1)
#m = np.diag(A,k=0)
#m = np.diag(A,k=1)
#m = np.diag(A,k=-1)
#m = np.diag(A,k=50)
#m = np.diag(A,k=-50)
#print(np.linalg.norm(A-B))
#print(m)
#plt.plot(m)
#plt.plot(np.diag(A,k=-1))
#plt.show()

#print(np.linalg.norm(B))