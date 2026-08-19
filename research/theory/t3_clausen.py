"""
T3 — singularity-free evaluator for the (L1) domination gap.

Delta(eta) = int ln|eta - xi(t)| dP_D(t)  -  (D/2) ln(1+eta^2),   xi(t)=tan(pi t/2L)
             (P_D = drop measure of total mass D)
and  eps''(x) = -Delta(xi(x))/pi.

Method: treat the drop measure as piecewise constant on grid cells and integrate the
LOG KERNEL EXACTLY over each cell.  Every antiderivative needed is a Clausen function
Cl2, and Cl2 is evaluated EXACTLY on the uniform grid theta_k = 2 pi k / M by aliasing
the Fourier series sum_n sin(n theta)/n^2 into a trigamma-weighted DFT:

    Cl2(2 pi k/M) = sum_{r=1}^{M-1} sin(2 pi r k/M) * psi_1(r/M)/M^2      (exact)

Consequences: no evaluation of ln at a singular point anywhere, no tan() overflow at
the seam, error O(h^2) from the piecewise-constant measure only.
"""
import numpy as np
from scipy.special import polygamma

_CACHE = {}

def _clausen_grid(M):
    """Cl2(2 pi k / M) for k = 0..M-1, exact to machine precision."""
    if M in _CACHE:
        return _CACHE[M]
    c = np.zeros(M)
    r = np.arange(1, M)
    c[1:] = polygamma(1, r / M) / M**2
    cl = -np.imag(np.fft.fft(c))
    _CACHE[M] = cl
    return cl

def gap_clausen(e_re, x):
    """Return (eps''_dom(x), D). eps''_dom = -Delta/pi evaluated on the same grid."""
    N = len(x); M = N - 1
    L = 0.5 * (x[-1] - x[0]); h = (x[-1] - x[0]) / M
    cl = _clausen_grid(M)
    # A(k) = (L/pi)*[ W(theta_k) - theta_k ln2 ],  W = -Cl2,  theta_k = 2 pi k/M
    kk = np.arange(-(M + 1), M + 2)
    W = -cl[np.mod(kk, M)]
    A = (L / np.pi) * (W - (2 * np.pi * kk / M) * np.log(2.0))
    dA = A[1:] - A[:-1]                      # dA[i] corresponds to k = kk[i+1]
    kdA = kk[1:]                             # k index of each dA entry
    off = -kdA[0]                            # dA_of(k) = dA[k + off]
    def dA_of(k):
        return dA[np.asarray(k) + off]

    w = e_re[:-1] - e_re[1:]                 # exact cell drop masses, len M
    D = w.sum()
    # C_j = dA_of(j+1), j = 0..M-1
    const = np.dot(w / h, dA_of(np.arange(1, M + 1)))
    # sum_j w_j/h * S_j(m),  S_j(m) = dA_of(m-j)  -> correlation
    ker = dA_of(np.arange(-(M - 1), M + 1))  # k from -(M-1) to M  (len 2M)
    conv = np.convolve(w / h, ker)           # index p -> sum_j (w/h)_j * ker[p-j]
    # need m-j = k  => ker index (m-j)+(M-1); conv index p = j + (m-j+M-1) = m+M-1
    S = conv[(M - 1):(M - 1) + N]
    Delta = S - const
    return -Delta / np.pi, D

if __name__ == '__main__':
    import sys, os
    _ROOT = "/Users/kunal/Documents/Spatial KK Project"
    sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, 'theory'))
    sys.path.insert(0, os.path.join(_ROOT, 'theory', 'passivity_2026Aug'))
    # sanity: Cl2 special values
    M = 12000
    cl = _clausen_grid(M)
    k = M // 6      # theta = pi/3  -> Gieseking 1.0149416064096536
    print("Cl2(pi/3) =", cl[k], " ref 1.0149416064096536")
    k = M // 4      # theta = pi/2  -> Catalan 0.9159655941772190
    print("Cl2(pi/2) =", cl[k], " ref 0.9159655941772190")
