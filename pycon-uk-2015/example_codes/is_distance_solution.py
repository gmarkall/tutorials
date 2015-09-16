import time
import numpy as np
from numba import jit


# Reference:
#
# http://stackoverflow.com/questions/28970883/inter-segment-distance-using-numba-jit-python/28977230#28977230

# Note that this solution provides a "starting point" for further optimisations
# rather than a maximally-optimised solution. See the StackOverflow thread for
# an alternative approach.
#
# This implementation uses memory allocation in nopython mode, which is a new
# feature for Numba 0.19.

# -----------------------------------------------------------------------------
# Modify the functions dot3d, and compute to use Numba and make them run as
# fast as possible.
# -----------------------------------------------------------------------------


@jit(nopython=True)
def dot3d(a,b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


@jit(nopython=True)
def compute(a):
    N = len(a)
    con_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):

            p0 = a[i,0:3]
            p1 = a[i,3:6]
            q0 = a[j,0:3]
            q1 = a[j,3:6]

            p0mq0 = np.zeros(p0.shape)
            for ii in range(len(p0)):
                p0mq0[ii] = p0[ii] - q0[ii] 
            p1mp0 = np.zeros(p1.shape)
            for ii in range(len(p0)):
                p1mp0[ii] = p1[ii] - p0[ii] 
            q1mq0 = np.zeros(q1.shape)
            for ii in range(len(p0)):
                q1mq0[ii] = q1 [ii]- q0[ii]
            

            s = ( dot3d((p1mp0),(q1mq0))*dot3d((q1mq0),(p0mq0)) - dot3d((q1mq0),(q1mq0))*dot3d((p1mp0),(p0mq0))) \
              / ( dot3d((p1mp0),(p1mp0))*dot3d((q1mq0),(q1mq0)) - dot3d((p1mp0),(q1mq0))**2 )
            t = ( dot3d((p1mp0),(p1mp0))*dot3d((q1mq0),(p0mq0)) - dot3d((p1mp0),(q1mq0))*dot3d((p1mp0),(p0mq0))) \
              / ( dot3d((p1mp0),(p1mp0))*dot3d((q1mq0),(q1mq0)) - dot3d((p1mp0),(q1mq0))**2 )

            p1mp0s = np.empty(p1mp0.shape)
            for ii in range(len(p1mp0)):
                p1mp0s[ii] = p1mp0[ii] * s
            q1mq0t = np.empty(q1mq0.shape)
            for ii in range(len(q1mq0)):
                q1mq0t[ii] = q1mq0[ii] * t
            tmp = 0.0
            for ii in range(len(p0)):
                tmp += ( (p0[ii] + p1mp0s[ii]) - (q0[ii] + q1mq0t[ii]) ) ** 2
            con_mat[i,j] = tmp

    return con_mat


# -----------------------------------------------------------------------------
# Reference implementation - do not modify this. It is used to compare the
# output and performance of your modified version.
# -----------------------------------------------------------------------------


def dot3d_ref(a,b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


def compute_reference(a):
    N = len(a)
    con_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):

            p0 = a[i,0:3]
            p1 = a[i,3:6]
            q0 = a[j,0:3]
            q1 = a[j,3:6]

            s = ( dot3d_ref((p1-p0),(q1-q0))*dot3d_ref((q1-q0),(p0-q0)) - dot3d_ref((q1-q0),(q1-q0))*dot3d_ref((p1-p0),(p0-q0))) \
              / ( dot3d_ref((p1-p0),(p1-p0))*dot3d_ref((q1-q0),(q1-q0)) - dot3d_ref((p1-p0),(q1-q0))**2 )
            t = ( dot3d_ref((p1-p0),(p1-p0))*dot3d_ref((q1-q0),(p0-q0)) - dot3d_ref((p1-p0),(q1-q0))*dot3d_ref((p1-p0),(p0-q0))) \
              / ( dot3d_ref((p1-p0),(p1-p0))*dot3d_ref((q1-q0),(q1-q0)) - dot3d_ref((p1-p0),(q1-q0))**2 )

            con_mat[i,j] = np.sum( (p0+(p1-p0)*s-(q0+(q1-q0)*t))**2 ) 

    return con_mat

# -----------------------------------------------------------------------------
# For execution of the test
# -----------------------------------------------------------------------------

def main(*args):
    iterations = 10
    if len(args) >= 1:
        iterations = int(args[0])
    v = np.random.random( (150,6) )
    
    time0 = time.time()
    for i in range(iterations):
        optimised = compute(v)
    time1 = time.time()
    print("Optimised time: %f msec" % ((time1 - time0) / iterations * 1000))

    time0 = time.time()
    for i in range(iterations):
        reference = compute_reference(v)
    time1 = time.time()
    print("Reference time: %f msec" % ((time1 - time0) / iterations * 1000))

    np.testing.assert_allclose(optimised, reference)

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
