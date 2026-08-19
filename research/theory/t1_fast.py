"""Fast vectorised Clausen + arc potentials for the T1 search."""
import numpy as np
from scipy.special import zeta
PI = np.pi
_NT = 14
_C = np.array([zeta(2*n)/(n*(2*n+1)) for n in range(1, _NT+1)])[::-1]   # Horner, high->low

def cl2(theta):
    th = np.mod(np.asarray(theta, float), 2*PI)
    sgn = np.where(th > PI, -1.0, 1.0)
    th = np.where(th > PI, 2*PI - th, th)
    t = np.maximum(th, 1e-300)
    u = (t/(2*PI))**2
    s = np.zeros_like(t)
    for c in _C:                      # Horner in u:  sum c_n u^n  (n=1..NT)
        s = (s + c)*u
    return sgn*np.where(th > 0, t - t*np.log(t) + t*s, 0.0)

def phi_arc(theta, c, w):
    """theta: (...,T) or broadcastable; c,w broadcast."""
    return (cl2(theta - c - w) - cl2(theta - c + w))/(2*w)

def m1_arc(c, w):
    return np.log(np.cos((c-w)/2)/np.cos((c+w)/2))/w

def gain_batch(p1, q1, p2, q2, th):
    """Vectorised over configs (1-D arrays). Returns (gain, lam); gain=nan if unbalanceable."""
    p1,q1,p2,q2 = map(np.asarray,(p1,q1,p2,q2))
    c1,w1 = (p1+q1)/2, (q1-p1)/2
    c2,w2 = (p2+q2)/2, (q2-p2)/2
    ok = (w1 > 1e-7) & (w2 > 1e-7)
    a = np.where(ok, m1_arc(np.where(ok,c1,0), np.where(ok,w1,1)), 0.0)
    b = np.where(ok, m1_arc(np.where(ok,c2,0), np.where(ok,w2,1)), 0.0)
    den = a - b
    ok &= np.abs(den) > 1e-12
    lam = np.where(ok, -b/np.where(ok, den, 1.0), np.nan)
    ok &= (lam >= 0) & (lam <= 1)
    T = th[None, :]
    P  = (lam[:,None]*phi_arc(T, c1[:,None], w1[:,None])
          + (1-lam)[:,None]*phi_arc(T, c2[:,None], w2[:,None]))
    Pi = (lam*phi_arc(np.array(PI), c1, w1) + (1-lam)*phi_arc(np.array(PI), c2, w2))
    g = P.max(axis=1) - Pi
    return np.where(ok, g, -np.inf), lam
