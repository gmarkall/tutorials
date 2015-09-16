import time
import numpy as np
from numba import jit

# -----------------------------------------------------------------------------
# Modify the functions check_neighbour, and find_max_neighbours to use Numba
# and make them run as fast as possible.
# -----------------------------------------------------------------------------


@jit(nopython=True)
def check_neighbour(grid, i, j):
    if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
        return grid[i, j]
    else:
        return False

@jit(nopython=True)
def find_max_neighbours(grid):
    max_neighbours = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            neighbour_count = 0
            for i_offset in -1, 0, 1:
                for j_offset in -1, 0, 1:
                    if i_offset == 0 and j_offset == 0:
                        continue
                    elif check_neighbour(grid, i + i_offset, j + j_offset):
                        neighbour_count += 1
            max_neighbours = max(max_neighbours, neighbour_count)
    return max_neighbours


# -----------------------------------------------------------------------------
# Reference implementation - do not modify this. It is used to compare the
# output and performance of your modified version.
# -----------------------------------------------------------------------------


def cell_region_ref(grid, i, j):
    # Return view over the region if possible
    if 1 <= i < grid.shape[0] - 1 and 1 <= j < grid.shape[1] - 1:
        return grid[i-1:i+2, j-1:j+2]

    # If not, create clipped region
    region = np.zeros((3, 3), dtype=grid.dtype)
    for k, ii in enumerate(range(i-1, i+2)):
        if 0 <= ii < grid.shape[0]:
            for l, jj in enumerate(range(j-1, j+2)):
                if 0 <= jj < grid.shape[1]:
                    region[k, l] = grid[ii, jj]
    return region


def find_max_neighbours_reference(grid):
    max_neighbours = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            neighbour_count = np.sum(cell_region_ref(grid, i, j)) - grid[i, j]
            max_neighbours = max(max_neighbours, neighbour_count)
    return max_neighbours


# -----------------------------------------------------------------------------
# For execution of the test
# -----------------------------------------------------------------------------


def main(*args):
    iterations = 10
    N = 300
    if len(args) >= 1:
        iterations = int(args[0])

    grid = (np.random.uniform(size=N*N) > 0.5).reshape((N, N))
    
    time0 = time.time()
    for i in range(iterations):
        optimised = find_max_neighbours(grid)
    time1 = time.time()
    print("Optimised time: %f msec" % ((time1 - time0) / iterations * 1000))

    time0 = time.time()
    for i in range(iterations):
        reference = find_max_neighbours_reference(grid)
    time1 = time.time()
    print("Reference time: %f msec" % ((time1 - time0) / iterations * 1000))

    assert optimised == reference


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
