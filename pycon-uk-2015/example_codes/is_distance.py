import time
import numpy as np


# Reference:
#
# http://stackoverflow.com/questions/28970883/inter-segment-distance-using-numba-jit-python/28977230#28977230


# -----------------------------------------------------------------------------
# Modify the functions dot3d, and compute to use Numba and make them run as
# fast as possible.
# -----------------------------------------------------------------------------


def dot3d(a,b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


def compute(a):
    N = len(a)
    con_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):

            p0 = a[i,0:3]
            p1 = a[i,3:6]
            q0 = a[j,0:3]
            q1 = a[j,3:6]

            s = ( dot3d((p1-p0),(q1-q0))*dot3d((q1-q0),(p0-q0)) - dot3d((q1-q0),(q1-q0))*dot3d((p1-p0),(p0-q0))) \
              / ( dot3d((p1-p0),(p1-p0))*dot3d((q1-q0),(q1-q0)) - dot3d((p1-p0),(q1-q0))**2 )
            t = ( dot3d((p1-p0),(p1-p0))*dot3d((q1-q0),(p0-q0)) - dot3d((p1-p0),(q1-q0))*dot3d((p1-p0),(p0-q0))) \
              / ( dot3d((p1-p0),(p1-p0))*dot3d((q1-q0),(q1-q0)) - dot3d((p1-p0),(q1-q0))**2 )

            con_mat[i,j] = np.sum( (p0+(p1-p0)*s-(q0+(q1-q0)*t))**2 ) 

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
