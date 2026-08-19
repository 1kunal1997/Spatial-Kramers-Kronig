"""E2 -- GAUGE CHECK.  Does production ht_derivative really realise the closed-form
kernel  (1/pi) ln[cos(pi t/2L)/|sin(pi(x-t)/2L)|]  including the cos numerator?

The cos(pi t/2L) numerator IS the DiHiTI gauge (the constant of integration fixed by
the detilt).  If it were absent or different, C25's whole a+b<=L threshold moves.
Test with a SINGLE off-centre atom, which has M1 != 0 -- so this probes the gauge
without the balance condition doing any work.
"""
import numpy as np
from common import *

L = 1.0
print(f"{'t0':>7} {'N':>8} {'max|err|':>12} {'rel rms':>12}  (excluding 0.05 nbhd of atom & seam)")
for t0 in (0.0, 0.3, -0.6, 0.85):
    for N in (1 << 14, 1 << 16, 40000):
        x, e_im = dihiti_of_atoms([(t0, 1.0)], N=N, L=L)
        ex = exact_kernel(x, [(t0, 1.0)], L)
        m = (np.abs(x - t0) > 0.05) & (np.abs(x) < 0.95 * L)
        err = np.abs(e_im[m] - ex[m])
        rel = np.sqrt(np.mean(err**2)) / np.sqrt(np.mean(ex[m]**2))
        print(f"{t0:7.2f} {N:8d} {err.max():12.3e} {rel:12.3e}")

# control: what if the gauge had NO cos numerator?
print("\nCONTROL -- same comparison against the kernel WITHOUT the cos(pi t/2L) factor:")
for t0 in (0.3, 0.85):
    x, e_im = dihiti_of_atoms([(t0, 1.0)], N=1 << 16, L=L)
    bad = -np.log(np.abs(np.sin(np.pi * (x - t0) / (2 * L)))) / np.pi
    m = (np.abs(x - t0) > 0.05) & (np.abs(x) < 0.95 * L)
    print(f"  t0={t0}: max|err| = {np.abs(e_im[m]-bad[m]).max():.4e}"
          f"   (correct-gauge offset would be ln cos = {np.log(np.cos(np.pi*t0/2))/np.pi:.4e})")
