"""The continuation is PERIODIC, not clamped. Redo the analysis accordingly.

scipy.signal.hilbert is FFT-based => it computes the CONJUGATE FUNCTION on a
circle of circumference 2L (cotangent kernel), not the line Hilbert transform
(1/(x-t) kernel). Consequences to test:

  (1) A profile with eps'(-L) != eps'(+L) has a JUMP under periodic wrap.
      The conjugate function of a jump is LOG-DIVERGENT. So the direct HT
      should show gain at the domain edge that grows like ln(N) under grid
      refinement -- a genuine singularity, not a fixed-size artifact.

  (2) np.gradient does NOT wrap, so DiHiTI never sees the jump. Claim:
      DiHiTI == (subtract the linear endpoint interpolant) + (periodic HT).
      Test by constructing that explicitly and comparing.

  (3) If (2) holds, DiHiTI's output is INDEPENDENT of the endpoint mismatch,
      while the true periodic conjugate function depends on it strongly.
"""
import sys
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project")
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
import tmm_helper as tmm_h

nb, L = 1.7, 2.5


def dihiti(x_, e):
    n = len(e)
    v = np.imag(hilbert(np.gradient(e, x_)))
    c = cumulative_trapezoid(v, x_, initial=0)
    return c - np.linspace(c[0], c[-1], n)


def direct(e):
    return np.imag(hilbert(e))


def deramped_direct(x_, e):
    """Subtract the straight line joining the endpoints, then periodic HT."""
    n = len(e)
    ramp = np.linspace(e[0], e[-1], n)
    return np.imag(hilbert(e - ramp))


print("=" * 78)
print("(1) Is the edge gain a genuine log singularity? Refine the grid.")
print("=" * 78)
print(f"{'N':>8} {'min eps'' (direct)':>20} {'argmin position':>18} {'jump at wrap':>14}")
for n in (1024, 2048, 4096, 8192, 16384, 32768):
    x_ = np.linspace(-L, L, n)
    e = tmm_h.logistic(x_, 4.0, nb)
    d = direct(e)
    i = int(np.argmin(d))
    where = 'LEFT edge' if i < n * 0.05 else ('RIGHT edge' if i > n * 0.95 else f'interior i={i}')
    print(f"{n:>8} {d.min():>20.4f} {where:>18} {abs(e[-1]-e[0]):>14.4f}")

print()
print("  ln(N) scaling check: successive differences should be ~constant")
prev = None
for n in (1024, 2048, 4096, 8192, 16384, 32768):
    x_ = np.linspace(-L, L, n)
    m = direct(tmm_h.logistic(x_, 4.0, nb)).min()
    if prev is not None:
        print(f"    N={n:>6}: min={m:>9.4f}   delta vs previous doubling = {m-prev:>8.4f}")
    prev = m

print()
print("=" * 78)
print("(2) Is DiHiTI equivalent to de-ramp + periodic HT?")
print("=" * 78)
N = 8192
xx = np.linspace(-L, L, N)


def gauss(x_, x0, s):
    return np.exp(-((x_ - x0) / s)**2)


A = nb**2 - 1.0
cases = [
    ('logistic k=4          (jump 1.89)', tmm_h.logistic(xx, 4.0, nb)),
    ('logistic + centred bump',           tmm_h.logistic(xx, 4.0, nb) + 1.0 * A * gauss(xx, 0, .6)),
    ('opp bumps a=1.0',                   tmm_h.logistic(xx, 4.0, nb)
                                          + 1.0 * (A * gauss(xx, -.9, .45) - A * gauss(xx, .9, .45))),
    ('triangle s=2.0        (no jump)',   1.0 + 2.0 * (np.sqrt(L**2 + .15**2) - np.sqrt(xx**2 + .15**2))),
]
print(f"{'profile':<36} {'max|DiHiTI - deramped|':>24} {'rel. to span':>14}")
for lb, e in cases:
    a, b = dihiti(xx, e), deramped_direct(xx, e)
    # both are defined up to an additive constant; compare after centring
    d = (a - a.mean()) - (b - b.mean())
    print(f"{lb:<36} {np.abs(d).max():>24.3e} {np.abs(d).max()/(e.max()-e.min()):>14.2e}")

print()
print("=" * 78)
print("(3) Does DiHiTI ignore the endpoint jump that the true periodic HT sees?")
print("=" * 78)
base = tmm_h.logistic(xx, 4.0, nb)
print(f"{'added ramp s*x':>16} {'jump |e(L)-e(-L)|':>19} {'min eps'' DiHiTI':>18} "
      f"{'min eps'' direct(periodic)':>27}")
for s in (0.0, 0.5, 1.0, 2.0, -1.0):
    e = base + s * xx
    print(f"{s:>16.1f} {abs(e[-1]-e[0]):>19.4f} {dihiti(xx, e).min():>18.6f} "
          f"{direct(e).min():>27.4f}")
