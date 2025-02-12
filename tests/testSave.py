import numpy as np
from EigenProblem import PerOp_FD_Sparse
from BranchingRates import g_1
from BranchingRates import g_2
from BranchingRates import g_3
from scipy.sparse import linalg
import matplotlib.pyplot as plt

n = 100
#e = [1/np.sqrt(2), 1/np.sqrt(2)]
e = [0,1]
dim = 25
lamb = [i*.2 for i in range(dim) ]

Eig = np.empty(dim)


for i in range(dim):
    r = lamb[i]**2/2 + 6
    B = PerOp_FD_Sparse(n, e, lamb[i], g_3)
    b = linalg.eigs(B, 1 , sigma=r)
    Eig[i] = b[0][0]
    
fig, ax = plt.subplots(figsize=(6, 6))
#ax.title.set_text('e=(1/sqrt 2,1/sqrt 2)')
ax.title.set_text('e=(0,1)')
plt.xlim(0, 5)
plt.ylim(0, 5)
ax.plot(lamb, Eig)
ax.axis('equal')
plt.show()
plt.close(fig)

np.savez('testA.npz', lamb=lamb, Eig=Eig)