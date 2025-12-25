import numpy as np
import numba as nb


def explicit_diffusion(un, xn, tau, r, sigma):
    du_dx = np.gradient(un, xn, edge_order=2)
    d2u_dx2 = np.gradient(du_dx, xn, edge_order=2)
    A = (r - 0.5 * sigma ** 2) * du_dx + 0.5 * sigma ** 2 * d2u_dx2
    return np.exp(-r * tau) * (un + tau * A)


def get_option_price_explicit(xn, tn, r, sigma, is_euro, payoff, phi_l, phi_r):
    T = tn[-1]
    nt = len(tn)
    nx = len(xn)
    u = np.zeros((nt, nx))
    u[-1] = payoff(np.exp(xn))
    
    for t in range(nt - 2, -1, -1):
        tau = tn[t + 1] - tn[t]
        u[t] = explicit_diffusion(u[t + 1], xn, tau, r, sigma)
        u[t][0] = phi_l(tn[t], np.exp(xn[0]))
        u[t][-1] = phi_r(tn[t], np.exp(xn[-1]))
        if not is_euro:
            u[t] = np.maximum(u[t], payoff(np.exp(xn)))
    return u


@nb.njit
def tridiag_solver(A, B, C, F):
    n = len(A)
    alpha, beta, x = np.zeros((3, n))
    alpha[0] = -C[0] / B[0]
    beta[0] = F[0] / B[0]
    for i in range(n - 1):
        alpha[i + 1] = -C[i + 1] / (B[i + 1] + alpha[i] * A[i + 1])
        beta[i + 1] = (F[i + 1] - A[i + 1] * beta[i]) / (B[i + 1] + alpha[i] * A[i + 1])
    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]
    return x

def from_tridiag(A, B, C):
    mtr = np.diag(B) + np.diag(A[1:], -1) + np.diag(C[:-1], 1)
    return mtr

@nb.njit
def implicit_diffusion(un, xn, tau, r, sigma, u_left, u_right):
    gam = r - 0.5 * sigma ** 2
    h = xn[1] - xn[0]
    alpha = 0.5 * ( (sigma / h) ** 2 - gam /  h )
    beta = - (sigma / h) ** 2
    gamma = 0.5 * ( (sigma / h) ** 2 + gam /  h )
    
    nx = len(xn)
    
    A, B, C, F = np.zeros((4, nx))
    A[1:-1] = -tau * alpha
    B[1:-1] = (1 - tau * beta)
    C[1:-1] = -tau * gamma
    B[0] = 1
    B[-1] = 1
    
    F = np.exp(-r * tau) * un
    F[0] = u_left
    F[-1] = u_right
    
    un = tridiag_solver(A, B, C, F)
    return un 


def get_option_price_implicit(xn, tn, r, sigma, is_euro, payoff, phi_l, phi_r):
    T = tn[-1]
    nt = len(tn)
    nx = len(xn)
    u = payoff(np.exp(xn))
    
    for t in range(nt - 2, -1, -1):
        tau = tn[t + 1] - tn[t]
        u_left = phi_l(tn[t], np.exp(xn[0]))
        u_right = phi_r(tn[t], np.exp(xn[-1]))
        u = implicit_diffusion(u, xn, tau, r, sigma, u_left, u_right)
        if not is_euro:
            u = np.maximum(u, payoff(np.exp(xn)))
    return u


def _get_option_price_implicit(xn, tn, r, sigma, is_euro, payoff, phi_l, phi_r):
    T = tn[-1]
    nt = len(tn)
    nx = len(xn)
    res = []
    u = payoff(np.exp(xn))
    res.append(u)
    for t in range(nt - 2, -1, -1):
        tau = tn[t + 1] - tn[t]
        u_left = phi_l(tn[t], np.exp(xn[0]))
        u_right = phi_r(tn[t], np.exp(xn[-1]))
        u = implicit_diffusion(u, xn, tau, r, sigma, u_left, u_right)
        if not is_euro:
            u = np.maximum(u, payoff(np.exp(xn)))
        res.append(u)
    return np.array(res)[::-1]


def get_option_price_krank_nikolson(xn, tn, r, sigma, is_euro, payoff, phi_l, phi_r):
    T = tn[-1]
    nt = len(tn)
    nx = len(xn)
    u = payoff(np.exp(xn))
    
    for t in range(nt - 2, -1, -1):
        tau = tn[t + 1] - tn[t]
        w = explicit_diffusion(u, xn, 0.5 * tau, r, sigma)
        u_left = phi_l(tn[t], np.exp(xn[0]))
        u_right = phi_r(tn[t], np.exp(xn[-1]))
        w[0] = 0.5 * (u[0] + u_left)
        w[-1] = 0.5 * (u[-1] + u_right)
        u = implicit_diffusion(w, xn, 0.5 * tau, r, sigma, u_left, u_right)
        if not is_euro:
            u = np.maximum(u, payoff(np.exp(xn)))
    return u