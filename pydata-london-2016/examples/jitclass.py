from numba import jit, jitclass, float64, int32
import numpy as np

# Number of array elements
N = 1000

# Numpy structured type implementation
#
# Three coordinate values and an additional value indicating species type at
# each point (s)

dtype = [
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('s', np.int32)
]

aos = np.zeros(N, dtype)


# JIT Class SoA implementation

vector_spec = [
    ('N', int32),
    ('x', float64[:]),
    ('y', float64[:]),
    ('z', float64[:]),
    ('s', int32[:])
]

@jitclass(vector_spec)
class VectorSoA(object):
    def __init__(self, N):
        self.N = N
        self.x = np.zeros(N, dtype=np.float64)
        self.y = np.zeros(N, dtype=np.float64)
        self.z = np.zeros(N, dtype=np.float64)
        self.s = np.zeros(N, dtype=np.int32)

soa = VectorSoA(N)


# Example iterating over x with the AoS layout:

@jit(nopython=True)
def set_x_aos(v):
    for i in range(len(v)):
        v[i]['x'] = i

set_x_aos(aos)

# Example iterating over x with the SoA layout:

@jit(nopython=True)
def set_x_soa(v):
    for i in range(v.N):
        v.x[i] = i

set_x_soa(soa)


# Check the operations were conceptually similar
for i in range(soa.N):
    assert soa.x[i] == aos[i]['x']
