import numpy as np
from scipy import stats as sps

def sample_bm(tn, nsim, mu=0, sigma=1):
    nt = len(tn) - 1
    X = np.zeros((nt + 1, nsim))
    for i in range(nt):
        tau = tn[i + 1] - tn[i]
        X[i + 1] = X[i] + mu * tau + sigma * np.sqrt(tau) * np.random.randn(nsim)
    return X

def sample_gbm(tn, nsim, x0, mu, sigma, match=True):
    b = sample_bm(tn, nsim, mu - 0.5 * sigma ** 2, sigma)
    x = x0 * np.exp(b)
    if match:
        mean = x0 * np.exp((tn - tn[0]) * mu).reshape(-1, 1)
        std = mean * np.sqrt( np.exp(sigma ** 2 * (tn-tn[0])) - 1.0 ).reshape(-1, 1)
        x = (x - np.mean(x, axis=1, keepdims=True)) / (np.std(x, axis=1, keepdims=True) + 1e-10) * std + mean
        x[0] = x0
    return x

def get_BS_call_price(S, r, sigma, K, T):
    df = np.exp(-r * T)
    forw = S / df
    theta = np.sqrt(T) * sigma
    
    d1 = np.log(forw / K) / theta + 0.5 * theta
    d2 = np.log(forw / K) / theta - 0.5 * theta
    
    N1 = sps.norm.cdf(d1)
    N2 = sps.norm.cdf(d2)
    
    return S * N1 - df * K * N2


def get_BS_put_price(S, r, sigma, K, T):
    return get_BS_call_price(S, r, sigma, K, T) - (S - K * np.exp(-r * T))