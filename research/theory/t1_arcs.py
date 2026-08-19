"""
T1 — Open Problem 8.1 (skew tolerance).  Exact machinery.

CIRCLE FORM of (L1).  phi = pi t/L in (-pi,pi), seam at +-pi.  mu = normalised drop measure.
    eps''(phi) = (D/pi) [ Phi_mu(pi) - Phi_mu(theta) ],   Phi_mu(t) = int ln|2 sin((t-phi)/2)| dmu
    PASSIVE  <=>  Phi_mu(theta) <= Phi_mu(pi) for all theta   ("the seam is the global max of Phi")
Balance (necessary, C8/M1=0):   int tan(phi/2) dmu = 0.

KHINCHIN (layer cake): mu unimodal about phi_m  <=>  mu is a mixture of UNIFORM ARCS
[c-w, c+w] that contain phi_m.  With the two affine constraints (mass, M1) Dubins-Klee gives
ext = mixtures of AT MOST TWO such arcs.  (L2) then reduces the whole unimodal-balanced family
to a 2-arc check.

Closed forms used (both exact):
    Phi_arc(theta; c,w) = [Cl2(theta-c-w) - Cl2(theta-c+w)] / (2w)
    M1_arc(c,w)         = (1/w) ln[ cos((c-w)/2) / cos((c+w)/2) ]
"""
import numpy as np
from scipy.special import zeta

_Z = np.array([zeta(2*n) for n in range(1, 26)])
_N = np.arange(1, 26)

def cl2(theta):
    """Clausen Cl2(theta) = sum sin(n theta)/n^2, any real argument, ~1e-15."""
    th = np.mod(np.asarray(theta, float), 2*np.pi)
    sgn = np.where(th > np.pi, -1.0, 1.0)
    th = np.where(th > np.pi, 2*np.pi - th, th)          # now th in [0, pi]
    out = np.zeros_like(th)
    nz = th > 0
    t = th[nz]
    s = t - t*np.log(t)
    u = (t/(2*np.pi))**2
    # sum_n zeta(2n) t^{2n+1} / (n(2n+1)(2pi)^{2n})
    powu = np.ones_like(t)
    for n, z in zip(_N, _Z):
        powu = powu*u
        s = s + z*t*powu/(n*(2*n+1))
    out[nz] = s
    return sgn*out

def phi_arc(theta, c, w):
    """Log potential at theta of the uniform probability measure on the arc [c-w, c+w]."""
    theta = np.asarray(theta, float)
    if w < 1e-12:
        return np.log(np.abs(2*np.sin((theta-c)/2)))
    return (cl2(theta-c-w) - cl2(theta-c+w))/(2*w)

def m1_arc(c, w):
    """int tan(phi/2) d(uniform arc)."""
    if w < 1e-12:
        return np.tan(c/2)
    return np.log(np.cos((c-w)/2)/np.cos((c+w)/2))/w

# ------------------------------------------------------------------ self-tests
if __name__ == '__main__':
    print("Cl2(pi/3) =", cl2(np.pi/3), " ref 1.0149416064096536")
    print("Cl2(pi/2) =", cl2(np.pi/2), " ref 0.9159655941772190")
    print("Cl2(2pi/3)=", cl2(2*np.pi/3), " ref 0.6766277376064358")
    print("Cl2(pi)   =", cl2(np.pi),    " ref 0.0")
    print("Cl2(-0.7) =", cl2(-0.7), " (odd) ref", -cl2(0.7))
    # arc potential vs direct quadrature
    rng = np.random.default_rng(0)
    worst_p = worst_m = 0.0
    for _ in range(200):
        c = rng.uniform(-2.5, 2.5); w = rng.uniform(1e-3, min(np.pi-abs(c), 2.0))
        th = rng.uniform(-np.pi, np.pi)
        ph = np.linspace(c-w, c+w, 400001)
        q = np.trapezoid(np.log(np.abs(2*np.sin((th-ph)/2)) + 1e-300), ph)/(2*w)
        worst_p = max(worst_p, abs(q - phi_arc(th, c, w)))
        qm = np.trapezoid(np.tan(ph/2), ph)/(2*w)
        worst_m = max(worst_m, abs(qm - m1_arc(c, w)))
    print(f"arc potential vs quadrature: max abs diff {worst_p:.2e}")
    print(f"arc M1        vs quadrature: max abs diff {worst_m:.2e}")
