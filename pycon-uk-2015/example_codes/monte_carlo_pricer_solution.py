import numpy as np
from math import sqrt, exp
from timeit import default_timer as timer
import sys
from numba import jit, float64

# XXX
# XXX Needs more optimisation to be a good solution

# Input parameters


StockPrice = 20.83
StrikePrice = 21.50
Volatility = 0.021  #  per year
InterestRate = 0.20

Maturity = 5. / 12.

NumPath = 500000
NumStep = 200


# Validation parameters 


StockPriceRef = 22.64
StandardErrorRef = 0.000433
PaidOffRef = 1.14
OptionPriceRef = 1.049


# -----------------------------------------------------------------------------
# Modify the functions step, and monte_carlo_pricer to use Numba and make them
# run as fast as possible.
# -----------------------------------------------------------------------------

@jit(nopython=True)
def step(dt, price, c0, c1, noise):
    return price * np.exp(c0 * dt + c1 * noise)

@jit(nopython=True)
def monte_carlo_pricer(paths, dt, interest, volatility):
    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * np.sqrt(dt)

    for p in range(paths.shape[0]):
        for j in range(1, paths.shape[1]):
            price = paths[p, j - 1]
            noise = np.random.normal(0., 1.)
            paths[p, j] = step(dt, price, c0, c1, noise)


# -----------------------------------------------------------------------------
# For execution of the test
# -----------------------------------------------------------------------------


def main():
    paths = np.zeros((NumPath, NumStep + 1), order='F')
    paths[:, 0] = StockPrice
    DT = Maturity / NumStep

    ts = timer()
    monte_carlo_pricer(paths, DT, InterestRate, Volatility)
    te = timer()

    ST = paths[:, -1]
    PaidOff = np.maximum(paths[:, -1] - StrikePrice, 0)
    print('Result')
    fmt =('%20s: %s')
    StockPriceResult = np.mean(ST)
    StandardErrorResult = np.std(ST) / sqrt(NumPath)
    PaidOffResult = np.mean(PaidOff)
    print(fmt % ('stock price', StockPriceResult))
    print(fmt % ('standard error', StandardErrorResult))
    print(fmt % ('paid off', PaidOffResult))
    optionprice = np.mean(PaidOff) * exp(-InterestRate * Maturity)
    print(fmt % ('option price', optionprice))

    print('Performance')
    NumCompute = NumPath * NumStep
    print(fmt % ('Mstep/second', '%.2f' % (NumCompute / (te - ts) / 1e6)))
    print(fmt % ('time elapsed', '%.3fs' % (te - ts)))

    # Some of these tolerances may need adjusting if they turn out to be a
    # little too tight.

    np.testing.assert_allclose(StockPriceResult, StockPriceRef, rtol=5e-4)
    np.testing.assert_allclose(StandardErrorResult, StandardErrorRef, rtol=5e-2)
    np.testing.assert_allclose(PaidOffResult, PaidOffRef, rtol=5e-2)
    np.testing.assert_allclose(optionprice, OptionPriceRef, rtol=1e-3)

    if '--plot' in sys.argv:
        from matplotlib import pyplot
        pathct = min(NumPath, 100)
        for i in range(pathct):
            pyplot.plot(paths[i])
        print('Plotting %d/%d paths' % (pathct, NumPath))
        pyplot.show()


if __name__ == '__main__':
    main()
