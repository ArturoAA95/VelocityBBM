import numpy as np
from EigenProblem import PerOp_FD_Sparse
from BranchingRates import g_1
from BranchingRates import g_2
from BranchingRates import g_3
from scipy.sparse import linalg
import matplotlib.pyplot as plt

n = 100
#e = [1/np.sqrt(2), 1/np.sqrt(2)]

N = 4
dim = 25
angle = [i*(2*np.pi/N) for i in range(N)]
lamb = [i*.08 for i in range(dim) ]

Gamma = np.empty((N, dim))
 
for j in range(N):
    print('e')
    e = [np.sin(angle[j]), np.cos(angle[j])]
    
    for i in range(dim):
        r = lamb[i]**2/2 + 1
        B = PerOp_FD_Sparse(n, e, lamb[i], g_2)
        b = linalg.eigs(B, 1 , sigma=r, return_eigenvectors=False)
        Gamma[j,i] = b[0]
    
    print('b')
    name = '{}'.format(j)
    np.savez('{0}.npz'.format(name), lamb=lamb, Eig=Gamma[j,:])
    print('a')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.title.set_text('{}_{}'.format(e[0], e[1]))
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    ax.plot(lamb, Gamma[j,:])
    ax.axis('equal')
    fig.savefig('{0}.png'.format(name))   # save the figure to file
    print('c')
    plt.close(fig)
    print('d')
    


