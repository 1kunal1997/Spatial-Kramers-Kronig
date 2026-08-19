import numpy as np
import scipy
import scipy.signal
import scipy.integrate
import timeit
import json, sys

nb = 1.7
L = 1.0
A = nb**2 - 1.0

# ---------------- profile family definitions ----------------

def logistic_u(x, w):
    return 1 + A/(1+np.exp(x/w))

def p2_u(x, alpha):
    base = logistic_u(x, 0.05)
    bump = alpha*A*(np.exp(-((x+0.45)/0.22)**2) - np.exp(-((x-0.45)/0.22)**2))
    return base + bump

def p3_u(x):
    return 1 + 2.0*(np.sqrt(L**2+0.15**2) - np.sqrt(x**2+0.15**2))

def p4_u(x, seed):
    rng = np.random.RandomState(seed)
    ncomp = rng.randint(3,7)
    centres = rng.uniform(-0.9*L, 0.9*L, ncomp)
    widths = rng.uniform(0.02*L, 0.3*L, ncomp)
    amps = rng.uniform(0.1, 1.0, ncomp)
    g = np.zeros_like(x)
    for c,wid,amp in zip(centres,widths,amps):
        g += amp*np.exp(-((x-c)/wid)**2)
    G = scipy.integrate.cumulative_trapezoid(g, x, initial=0.0)
    u = 1 + A*(1 - (G[-1]-G)/ (G[-1]-G[0]) )  # placeholder, fixed below
    return None  # unused, real construction in build_p4

def build_p4(x, seed):
    rng = np.random.RandomState(seed)
    ncomp = rng.randint(3,7)
    centres = rng.uniform(-0.9*L, 0.9*L, ncomp)
    widths = rng.uniform(0.02*L, 0.3*L, ncomp)
    amps = rng.uniform(0.1, 1.0, ncomp)
    g = np.zeros_like(x)
    for c,wid,amp in zip(centres,widths,amps):
        g += amp*np.exp(-((x-c)/wid)**2)
    G = scipy.integrate.cumulative_trapezoid(g, x, initial=0.0)
    u_raw = 1 + (G[-1] - G)   # decreasing, nonneg drop
    # rescale so u[0]-u[-1] = A
    drop = u_raw[0]-u_raw[-1]
    if drop <= 0:
        drop = 1e-12
    u = 1 + A*(u_raw-u_raw[-1])/drop
    return u

def profiles_at_N(N):
    """Return dict name->u for given N, using x=np.linspace(-L,L,N,endpoint=True)."""
    x = np.linspace(-L, L, N, endpoint=True)
    profs = {}
    for w in [0.15, 0.05, 0.02, 0.005]:
        profs[f"P1_logistic_w{w}"] = logistic_u(x, w)
    for alpha in [0.0, 0.3, 0.6, 0.8, 0.9, 1.0, 1.5]:
        profs[f"P2_bumps_a{alpha}"] = p2_u(x, alpha)
    profs["P3_triangle"] = p3_u(x)
    for seed in range(20):
        profs[f"P4_random_s{seed}"] = build_p4(x, seed)
    return x, profs

# ---------------- pipelines ----------------

def dihiti(u, x):
    du = np.gradient(u, x)
    v  = np.imag(scipy.signal.hilbert(du))
    e  = scipy.integrate.cumulative_trapezoid(v, x, initial=0.0)
    return e - np.linspace(e[0], e[-1], len(x))

def onehilb(u, x):
    Lh = 0.5*(x[-1]-x[0]); D = u[0]-u[-1]
    p = u + (D/(2*Lh))*x
    v = np.imag(scipy.signal.hilbert(p))
    return v - v[0]

def onehilb_rfft(u, x):
    Lh = 0.5*(x[-1]-x[0]); D = u[0]-u[-1]
    p = u + (D/(2*Lh))*x
    N = len(p)
    P = np.fft.rfft(p)
    # build Hilbert multiplier for rfft representation matching scipy.signal.hilbert
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = 1
        h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2
    # scipy.signal.hilbert uses full fft; replicate via rfft/irfft:
    # analytic signal a = ifft(fft(p)*h_full); imag(a) is what we want.
    # Using rfft: Xf = rfft(p) corresponds to fft(p)[0:N//2+1].
    # h on rfft bins: h[0]=1 (and h[N/2]=1 if N even, no doubling), interior *2, then irfft with proper scaling
    if N % 2 == 0:
        mult = np.ones(N//2+1)
        mult[1:N//2] = 2.0
        mult[N//2] = 1.0
    else:
        mult = np.ones((N+1)//2)
        mult[1:] = 2.0
    Xf = P*mult
    # analytic signal real part via irfft won't directly give complex analytic signal.
    # Instead compute imaginary part directly: for real input, analytic signal a(t) = p(t) + i*Hilbert(p)(t)
    # a = ifft( fft(p) * h_full ), h_full as above (2 on positive freqs excluding dc/nyquist, 0 on negative, 1 on dc/nyquist)
    # We can get this via irfft of a modified spectrum using the fact that a is NOT conjugate symmetric,
    # so we must do full complex ifft, but we can build it from rfft output only (Hermitian half) since a's
    # rfft-visible half already contains all needed info combined with construction:
    # a_full_freq[k] = h[k]*P[k] for k=0..N-1 (P from full fft). For k in [0, N/2], P[k]=Xf[k] (rfft).
    # For k in (N/2, N-1], P[k] = conj(Xf[N-k]).
    Yf = np.zeros(N, dtype=complex)
    if N % 2 == 0:
        Yf[0] = Xf[0]
        Yf[N//2] = Xf[N//2]
        Yf[1:N//2] = 2*P[1:N//2]
        # negative freq bins zero
    else:
        Yf[0] = Xf[0]
        Yf[1:(N+1)//2] = 2*P[1:(N+1)//2]
    a = np.fft.ifft(Yf)
    v = a.imag
    return v - v[0]

def sign_min_interior(vals, x, thresh=0.99*L):
    mask = np.abs(x) < thresh
    m = np.min(vals[mask])
    return np.sign(m), m

if __name__ == "__main__":
    pass
