import numpy as np
import time


RISKFREE = 0.02
VOLATILITY = 0.30


# -----------------------------------------------------------------------------
# Modify the functions cnd, and black_scholes to use Numba and make them run as
# fast as possible.
# -----------------------------------------------------------------------------


def cnd(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438
    K = 1.0 / (1.0 + 0.2316419 * np.abs(d))
    ret_val = (RSQRT2PI * np.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    return np.where(d > 0, 1.0 - ret_val, ret_val)


def black_scholes(callResult, putResult, stockPrice, optionStrike, optionYears,
                  Riskfree, Volatility):
    S = stockPrice
    X = optionStrike
    T = optionYears
    R = Riskfree
    V = Volatility
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd(d1)
    cndd2 = cnd(d2)

    expRT = np.exp(- R * T)
    callResult[:] = (S * cndd1 - X * expRT * cndd2)
    putResult[:] = (X * expRT * (1.0 - cndd2) - S * (1.0 - cndd1))


# -----------------------------------------------------------------------------
# Reference implementation - do not modify this. It is used to compare the
# output and performance of your modified version.
# -----------------------------------------------------------------------------


def cnd_reference(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438
    K = 1.0 / (1.0 + 0.2316419 * np.abs(d))
    ret_val = (RSQRT2PI * np.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    return np.where(d > 0, 1.0 - ret_val, ret_val)


def black_scholes_reference(callResult, putResult, stockPrice, optionStrike,
                            optionYears, Riskfree, Volatility):
    S = stockPrice
    X = optionStrike
    T = optionYears
    R = Riskfree
    V = Volatility
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd_reference(d1)
    cndd2 = cnd_reference(d2)

    expRT = np.exp(- R * T)
    callResult[:] = (S * cndd1 - X * expRT * cndd2)
    putResult[:] = (X * expRT * (1.0 - cndd2) - S * (1.0 - cndd1))


# -----------------------------------------------------------------------------
# For execution of the test
# -----------------------------------------------------------------------------


def randfloat(rand_var, low, high):
    return (1.0 - rand_var) * low + rand_var * high


def main (*args):
    OPT_N = 4000000
    iterations = 10
    if len(args) >= 1:
        iterations = int(args[0])
    
    stockPrice = randfloat(np.random.random(OPT_N), 5.0, 30.0)
    optionStrike = randfloat(np.random.random(OPT_N), 1.0, 100.0)
    optionYears = randfloat(np.random.random(OPT_N), 0.25, 10.0)

    callResultOptimised = np.zeros(OPT_N)
    putResultOptimised = -np.ones(OPT_N)
    
    time0 = time.time()
    for i in range(iterations):
        black_scholes(callResultOptimised, putResultOptimised, stockPrice,
                      optionStrike, optionYears, RISKFREE, VOLATILITY)
    time1 = time.time()
    print("Optimised time: %f msec" % ((time1 - time0) / iterations * 1000))

    callResultReference = np.zeros(OPT_N)
    putResultReference = -np.ones(OPT_N)
    
    time0 = time.time()
    for i in range(iterations):
        black_scholes_reference(callResultReference, putResultReference,
                                stockPrice, optionStrike, optionYears,
                                RISKFREE, VOLATILITY)
    time1 = time.time()
    print("Reference time: %f msec" % ((time1 - time0) / iterations * 1000))

    delta = np.abs(callResultOptimised - callResultReference)
    L1norm = delta.sum() / np.abs(callResultReference).sum()
    print("L1 norm: %E" % L1norm)
    print("Max absolute error: %E" % delta.max())
    np.testing.assert_allclose(callResultOptimised, callResultReference)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
