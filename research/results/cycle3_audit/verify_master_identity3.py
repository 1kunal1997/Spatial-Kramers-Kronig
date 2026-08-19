"""High-precision mpmath reference for eps''_true(x) via the C1 master identity,
analytic logistic g(t), to nail down whether gap_clausen (t3_clausen.py) is really
O(h^2) or actually only O(h)."""
import sys, numpy as np
import mpmath as mp
mp.mp.dps = 40
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project")
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/theory")
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/research/theory")
import tmm_helper as tmm_h
from t3_clausen import gap_clausen

NB, k = mp.mpf('1.7'), mp.mpf('4.0')
L = mp.mpf('2.5')
A = NB**2 - 1

def g(t):
    e = mp.e**(k*t)
    return A*k*e/(1+e)**2

def kernel_true(t, xpt, L):
    return mp.log(abs(mp.cos(mp.pi*t/(2*L)) / mp.sin(mp.pi*(xpt-t)/(2*L))))

def eps_im_true(xpt, L):
    xpt = mp.mpf(xpt)
    f = lambda t: g(t)*kernel_true(t, xpt, L)
    val = mp.quad(f, [-L, xpt, L])
    return val/mp.pi

test_x = [-1.5, -0.7, -0.2, 0.0, 0.3, 0.9, 1.7]
truth = {xp: eps_im_true(xp, L) for xp in test_x}
for xp,v in truth.items():
    print(f"  x={xp:+.2f}  truth={mp.nstr(v, 15)}")

import numpy as np
for N in [513, 1025, 2049, 4097, 8193]:
    x = np.linspace(-2.5, 2.5, N)
    e_re = tmm_h.logistic(x, 4.0, 1.7)
    dom, D = gap_clausen(e_re, x)
    errs = []
    for xp in test_x:
        idx = np.argmin(np.abs(x-xp))
        errs.append(abs(dom[idx]-float(truth[xp])))
    ratio = ''
    print(f"N={N:5d}  max err vs mpmath truth = {max(errs):.6e}  h={5.0/(N-1):.3e}")
