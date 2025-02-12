import numpy as np
from EigenProblem import PerOp_FD_Sparse
from BranchingRates import g_1
from scipy.sparse import linalg
import matplotlib.pyplot as plt

n = 100
e = [1/np.sqrt(2), 1/np.sqrt(2)]
dim = 25
lamb = [i*.1 for i in range(dim) ]

Eig = np.empty(dim)
#Eigv = np.empty((dim, n**2))


for i in range(dim):
    r = lamb[i]**2/2 + 1
    B = PerOp_FD_Sparse(n, e, lamb[i], g_1)
    b = linalg.eigs(B, 1 , sigma=r)
    Eig[i] = b[0][0]
    
fig, ax = plt.subplots(figsize=(6, 6))
ax.title.set_text('e=(1/sqrt 2,1/sqrt 2)')
ax.plot(lamb, Eig)
ax.axis('equal')
plt.show()
plt.close(fig)

np.savez('test2.npz', lamb=lamb, Eig=Eig)