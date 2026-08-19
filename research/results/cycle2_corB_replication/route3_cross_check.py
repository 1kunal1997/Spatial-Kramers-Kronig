"""
Route 3 (weaker cross-check): direct O(N^2) discrete Hilbert transform via explicit
cotangent-kernel summation, NOT using scipy's FFT (scipy.signal.hilbert). This checks that
scipy's hilbert() -- which tmm_helper.ht_derivative relies on -- is actually behaving as the
periodic/circular discrete Hilbert transform (DHT) that the mission's foundational assumption
1 claims (no unexpected padding/windowing/off-by-one).

Standard formula for the discrete Hilbert transform of a length-N periodic sequence u[n]
(N even): H[u][n] = sum_m h[n-m] u[m], with kernel
    h[k] = 0                       if k even
    h[k] = (2/N) * cot(pi k / N)   if k odd
This is derived from the same "H[e^{2pi i k n/N}] = -i sgn(k) e^{...}" mode rule used in route 1,
applied to a length-N DFT instead of a continuum Fourier series -- i.e. it is the discrete analog,
independently coded here as an explicit double loop / matrix multiply (O(N^2)), not via FFT.
"""
import numpy as np
from scipy.signal import hilbert as scipy_hilbert

def direct_discrete_hilbert(u):
    N = len(u)
    assert N % 2 == 0, "use even N for the standard DHT kernel"
    k = np.arange(N)
    # kernel h[k], k=0..N-1 (k=0 excluded, and even k are 0)
    h = np.zeros(N)
    odd = (k % 2 == 1)
    h[odd] = (2.0 / N) / np.tan(np.pi * k[odd] / N)
    # circular convolution via direct O(N^2) matrix (NOT FFT)
    idx = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N
    H = h[idx]
    return H @ u

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for N in [64, 128, 256]:
        u = rng.normal(size=N)
        u -= u.mean()
        v_direct = direct_discrete_hilbert(u)
        v_scipy = np.imag(scipy_hilbert(u))
        err = np.max(np.abs(v_direct - v_scipy)) / np.max(np.abs(v_scipy))
        print(f"N={N}: max relative error between direct O(N^2) cotangent-kernel DHT "
              f"and scipy.signal.hilbert = {err:.3e}")
