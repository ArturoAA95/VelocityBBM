import numpy as np
from EigenProblem import PerOp_FD_Sparse
from BranchingRates import g_1
from BranchingRates import g_2
from BranchingRates import g_3
from scipy.sparse import linalg
import matplotlib.pyplot as plt

n = 200

N = 2
dim = 20
angle = [i*(np.pi/N) for i in range(N)]
lamb = [(i+5)*.08 for i in range(dim) ]

Gamma = np.empty((N, dim))
 
for j in range(N):
    e = [np.sin(angle[j]), np.cos(angle[j])]
    print(e)
    for i in range(dim):
        r = lamb[i]**2/2 + 6
        B = PerOp_FD_Sparse(n, e, lamb[i], g_3)
        b = linalg.eigs(B, 1 , return_eigenvectors=False)
        print(b)
        print(i)
        Gamma[j,i] = b[0]
    
    print(j)

print(np.linalg.norm(Gamma[0,:]-Gamma[1,:]))
x = [i**2+1 for i in lamb]
c_1 = [j/i for j,i in zip(Gamma[0,:], lamb)]
c_2 = [j/i for j,i in zip(Gamma[1,:], lamb)]

fig, ax = plt.subplots(figsize=(6, 6))
ax.title.set_text('{}_{}'.format(e[0], e[1]))
plt.xlim(0, 5)
plt.ylim(0, 5)
#ax.plot(lamb, Gamma[0,:])
#ax.plot(lamb, Gamma[1,:])
ax.plot(lamb, c_1)
ax.plot(lamb, c_2)
ax.axis('equal')
plt.show()
#fig.savefig('{}.png'.format(j))   # save the figure to file
plt.close(fig)

#np.savez('test.npz', lamb=lamb, Gamma=Gamma)

    


