# Numba Exercise 2.3 - Loop Lifting
#
# gauss2d computes a Gaussian function. Ideally we would run it in nopython
# mode, but this is not possible because the np.meshgrid function is not
# supported (try running it!)
#
# In order to get this code to execute, remove the argument nopython=True from
# the jit decorator.
#
# Although the meshgrid function prevents nopython mode compilation, object mode
# can successfully lift the inner loop, because it only uses supported types and
# functions. To observe this, run from the command line:
#
# $ numba --annotate-html gauss.html Numba_Exercise_2_3.py
#
# Now, open up gauss.html in a web browser. You will see there are several
# variables and functions highlighted in red, indicating their incompatibility
# with nopython mode.
#
# However, the loop is highlighted in green, indicating a successful loop lift. If
# you wish to see the complete Numba IR the corresponds to the code, click the
# “show Numba IR” text (although this doesn’t look like a clickable link, it is!).

import numpy as np
from numba import jit
import pylab
import math

MU = 0.0
THETA = 1.0

@jit
def gauss2d(x, y):
    x, y = np.meshgrid(x, y)
    grid = np.empty_like(x)

    a = 1.0 / (THETA * np.sqrt(2 * math.pi))

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            grid[i, j] = a * np.exp( - ( x[i,j]**2 / (2 * THETA) + y[i,j]**2 / (2 * THETA) ) )

    return x, y, grid


X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)

x, y, z = gauss2d(X, Y)

pylab.imshow(z)
pylab.show()
