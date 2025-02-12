import numpy as np
from scipy import sparse


def I_To_C(i, n):
    return [(i%n)/n, (np.floor(i/n))/n]

def C_To_I(x, y, n):
    pass


# e is a unitary vector, lambda is greater equal to 0,
# g is a periodic branching rate.
# returns matrix for n-approx finite difference periodic operator 
# Lu = 1/2 Delta u + lamb * e^T Nabla u + (1/2 lamb^2 + g) u


def PerOp_FD(n, e, lamb, g):
    #diagonal
    aux = -4/(2*n**2) - lamb*(e[0]+e[1])/n + .5*lamb**2 
    dim = n*n

    A = (aux)*np.identity(dim)
    for i in range(dim):
        coord = I_To_C(i, n)
        #print(coord,g(coord[0], coord[1]))
        A[i,i] += g(coord[0], coord[1])
    # Right and Up
        A[i, (i+n)%dim] += 1/(2*n**2) + lamb*e[1]/n
        A[i, (i+1)%dim] += 1/(2*n**2) + lamb*e[0]/n
    # Left and down
        A[i, (i-n)%dim] += 1/(2*n**2)
        A[i, (i-1)%dim] += 1/(2*n**2)

    return A

#Same function as above but with sparse matrix to 
#allow large discretizations

def PerOp_FD_Sparse(n, e, lamb, g):
    #diagonals
    dim = n*n 
    aux = -4/(2*n**2) - lamb*(e[0]+e[1])/n + .5*lamb**2 

    D_0 = aux*np.ones(dim)
    for i in range(dim):
        coord = I_To_C(i, n)
        D_0[i] += g(coord[0], coord[1])
    
    D_1 = (1/(2*n**2) + lamb*e[0]/n)*np.ones(dim-1)
    D_m1 = (1/(2*n**2))*np.ones(dim-1)

    D_n = (1/(2*n**2) + lamb*e[1]/n)*np.ones(dim-n)
    D_mn = (1/(2*n**2))*np.ones(dim-n)

    D_dimn = (1/(2*n**2))*np.ones(n)
    D_mdimn = (1/(2*n**2) + lamb*e[1]/n)*np.ones(n)

    D_dim = [1/(2*n**2)]
    D_mdim = [1/(2*n**2) + lamb*e[0]/n]

    A = sparse.diags([D_0, D_1, D_m1, D_n, D_mn, D_dimn, D_mdimn, D_dim, D_mdim], 
                     [0, 1, -1, n , -n, dim-n, -dim+n, dim-1, -dim+1], 
                     format = "dia")
    
    return A
