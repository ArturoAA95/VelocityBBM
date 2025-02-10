import numpy as np


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
    aux = -4/(2*n**2) - lamb*(e[1]+e[2])/n + .5*lamb**2 
    dim = n*n
    A = (aux)*np.identity(dim)
    
    for i in range(dim):
        coord = I_To_C(i, n)
        A[i,i] += g(coord[0], coord[1])
    # Right and Up
        A[i, (i+1)%n] += 1/(2*n**2) + lamb*e[0]/n
        A[(i+1)%n, i] += 1/(2*n**2) + lamb*e[1]/n
    # Left and down
        A[i, (i-1)%n] += 1/(2*n**2)
        A[(i-1)%n, i] += 1/(2*n**2)

    return A



