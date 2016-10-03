from numba import generated_jit, types
import numpy as np
import math

def scalar_1norm(x):
    '''Absolute value of x'''
    return math.fabs(x)

def vector_1norm(x):
    '''Sum of absolute values of x'''
    return np.sum(np.abs(x))

def matrix_1norm(x):
    '''Max sum of absolute values of columns of x'''
    colsums = np.zeros(x.shape[1])
    for i in range(len(colsums)):
        colsums[i] = np.sum(np.abs(x[:, i]))
    return np.max(colsums)

def bad_1norm(x):
    raise TypeError("Unsupported type for 1-norm")

@generated_jit(nopython=True)
def l1_norm(x):
    if isinstance(x, types.Number):
        return scalar_1norm
    if isinstance(x, types.Array) and x.ndim == 1:
        return vector_1norm
    elif isinstance(x, types.Array) and x.ndim == 2:
        return matrix_1norm
    else:
        return bad_1norm

M = 10
N = 5

x0 = np.random.rand()
x1 = np.random.rand(M)
x2 = np.random.rand(M * N).reshape(M, N)

# np.linalg.norm won't norm a scalar, but check our matrix and vector
# implementations agree
np.testing.assert_allclose(np.linalg.norm(x1, 1), l1_norm(x1))
np.testing.assert_allclose(np.linalg.norm(x2, 1), l1_norm(x2))

print("L1 norm of x0: %s" % l1_norm(x0))
print("L1 norm of x1: %s" % l1_norm(x1))
print(l1_norm(x2))

try:
    l1_norm(np.zeros((0,0,0)))
except TypeError as e:
    print("Norming 3-tensor failed: %s" % e.args[0])
