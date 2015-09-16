import numpy as np
import time
import math
from numba import jit

# Reference: https://jakevdp.github.io/blog/2012/08/24/numba-vs-cython/

# -----------------------------------------------------------------------------
# Modify the function pairwise to use Numba and make them run as fast as
# possible.
# -----------------------------------------------------------------------------


@jit(nopython=True)
def pairwise(X, D):
    M = X.shape[0]
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(X.shape[1]):
                dk = X[i,k] - X[j, k]
                d += dk * dk
            D[i, j] = np.sqrt(d)


# -----------------------------------------------------------------------------
# Reference implementation - do not modify this. It is used to compare the
# output and performance of your modified version.
# -----------------------------------------------------------------------------


def pairwise_reference(X, D):
    M = X.shape[0]
    for i in range(M):
        for j in range(M):
            D[i, j] = np.sqrt( np.sum(np.power( X[i,:] - X[j,:], 2)) )


# -----------------------------------------------------------------------------
# For execution of the test
# -----------------------------------------------------------------------------


def main(*args):
    iterations = 10
    N = 100
    if len(args) >= 1:
        iterations = int(args[0])

    X = np.random.random(N*N).reshape((N,N))
    D_python = np.empty_like(X)
    D_arrays = np.empty_like(X)

    time0 = time.time()
    for i in range(iterations):
        pairwise(X, D_python)
    time1 = time.time()
    print("Optimised time: %f msec" % ((time1 - time0) / iterations * 1000))
        
    time0 = time.time()
    for i in range(iterations):
        pairwise_reference(X, D_arrays)
    time1 = time.time()
    print("Reference time: %f msec" % ((time1 - time0) / iterations * 1000))
    
    np.testing.assert_allclose(D_arrays, D_python)
    

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
