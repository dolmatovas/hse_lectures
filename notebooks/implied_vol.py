import numpy as np
import numba as nb
import math

@nb.vectorize([nb.float64(nb.float64)])
def erf(x):
    return math.erf(x)


@nb.vectorize([nb.float64(nb.float64)])
def norm_cdf(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))


@nb.vectorize([nb.float64(nb.float64)])
def norm_pdf(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


@nb.njit
def BS(f, k, theta):
    d1 = np.log(f / k) / theta + 0.5 * theta
    d2 = d1 - theta
    pv = f * norm_cdf(d1) - k * norm_cdf(d2)
    return pv


@nb.njit
def BS_vega(f, k, theta):
    d1 = np.log(f / k) / theta + 0.5 * theta
    d2 = d1 - theta
    pv = f * norm_cdf(d1) - k * norm_cdf(d2)
    vega = f * norm_pdf(d1)
    return pv, vega


num_coefs = np.r_[  3.994961687345134e-1, 
            2.100960795068497e+1, 
            4.980340217855084e+1, 
            5.988761102690991e+2, 
            1.848489695437094e+3, 
            6.106322407867059e+3, 
            2.493415285349361e+4, 
            1.266458051348246e+4]

den_coefs = np.r_[  1.0, 
            4.990534153589422e+1, 
            3.093573936743112e+1, 
            1.495105008310999e+3, 
            1.323614537899738e+3, 
            1.598919697679745e+4, 
            2.392008891720782e+4, 
            3.608817108375034e+3, 
            -2.067719486400926e+2, 
            1.174240599306013e+1]


@nb.njit
def horner(x, coefs):
    res = np.zeros_like(x)
    for a in coefs[::-1]:
        res = res * x + a
    return res


@nb.njit
def get_norm_iv_approx(f, k, c, t):
    """
        very good approximation for normal implied vol
        A Black-Scholes user's guide to the Bachelier model
    """
    v = np.abs(f - k + (f == k) * 1e-8) / (2 * c - (f - k))
    eta = v / np.arctanh(v)
    num = horner(eta, num_coefs)
    den = horner(eta, den_coefs)
    h = np.sqrt(eta) * num / den
    return np.sqrt(np.pi / (2 * t)) * (2 * c - (f - k)) * h


@nb.njit
def norm_to_log(iv, f, k, t):
    x = k / f
    num = 1 + iv ** 2 * t / 24 / x / f ** 2
    den = f * np.sqrt(x) * (1 + np.log(x) ** 2 / 24)
    return iv * num / den


@nb.njit
def get_lognorm_iv_approx(f, k, c, t):
    """
        very good approximation for lognormal implied vol
        A Black-Scholes user's guide to the Bachelier model
    """
    iv = get_norm_iv_approx(f, k, c, t)
    return norm_to_log(iv, f, k, t)


@nb.njit
def get_implied_vol(prices, f, k, t, verbose=False):
    theta = np.sqrt(t) * get_lognorm_iv_approx(f, k, prices, t)
    # initial guess
    # theta = np.sqrt(2 * np.abs(np.log(f / k))
    # newton algorithm
    for i in range(15):
        pv, vega = BS_vega(f, k, theta)
        theta -= (pv - prices) / vega
        if verbose:
            print(i, "step, error =", np.linalg.norm(pv - prices))
    print('Final residual:', np.linalg.norm(pv - prices))
    return theta / np.sqrt(t)