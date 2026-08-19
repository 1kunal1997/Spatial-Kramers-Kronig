"""M3: grid convergence of the PASS->FAIL threshold alpha* for the
'opposite bumps' family, at N = 2**11 ... 2**16.

u(x) = logistic(x, k=20) + alpha*A*(exp(-((x+0.45)/0.22)^2) - exp(-((x-0.45)/0.22)^2))
       A = 1.89, nb = 1.7   (matches pg.A_BUMP = nb**2 - 1)

We use interior-min (|x| <= 0.95 L) as the passivity criterion, per task spec.
L = 2.5 (native generator domain, consistent with M1/M2).
"""
import numpy as np
from common import pg, tmm_h, grid, dihiti, OUTDIR
import os

L = 2.5
NB = 1.7
A_BUMP = NB**2 - 1  # = 1.89, matches pg.A_BUMP


def u_of(x, k, alpha, sep=0.45, w=0.22):
    base = tmm_h.logistic(x, k, NB)
    bump = alpha * A_BUMP * (np.exp(-((x + sep) / w)**2) - np.exp(-((x - sep) / w)**2))
    return base + bump


def interior_min(x, e_im, L_, frac=0.95):
    mask = np.abs(x) <= frac * L_
    return float(e_im[mask].min())


def find_alpha_star(N, k=20.0, alo=0.5, ahi=1.2, tol=1e-4):
    """Bisection for the alpha where interior_min crosses 0 (PASS->FAIL)."""
    x = grid(L, N)

    def im_at(alpha):
        e_re = u_of(x, k, alpha)
        e_im = dihiti(e_re, x)
        return interior_min(x, e_im, L)

    # first confirm alo passive, ahi failing; if not, scan
    f_lo = im_at(alo)
    f_hi = im_at(ahi)
    if not (f_lo >= 0 and f_hi < 0):
        # coarse scan to bracket the sign change
        grid_alpha = np.linspace(alo, ahi, 71)
        vals = [im_at(a) for a in grid_alpha]
        sign_change = None
        for i in range(len(vals) - 1):
            if vals[i] >= 0 and vals[i + 1] < 0:
                sign_change = (grid_alpha[i], grid_alpha[i + 1])
                break
        if sign_change is None:
            return None, f_lo, f_hi
        alo, ahi = sign_change
        f_lo, f_hi = im_at(alo), im_at(ahi)

    while (ahi - alo) > tol:
        amid = 0.5 * (alo + ahi)
        fmid = im_at(amid)
        if fmid >= 0:
            alo, f_lo = amid, fmid
        else:
            ahi, f_hi = amid, fmid
    return 0.5 * (alo + ahi), f_lo, f_hi


Ns = [2**11, 2**12, 2**13, 2**14, 2**15, 2**16]
results = []
for N in Ns:
    astar, f_lo, f_hi = find_alpha_star(N)
    results.append((N, astar))
    print(f"N=2^{int(np.log2(N))} = {N:>6}   alpha* = {astar}")

alphas = np.array([r[1] for r in results if r[1] is not None])
spread = alphas.max() - alphas.min()
rel_spread = spread / alphas.mean()

print(f"\nalpha* values: {alphas}")
print(f"spread (max-min) = {spread:.6e}   mean = {alphas.mean():.6f}   relative spread = {rel_spread:.4%}")

with open(os.path.join(OUTDIR, 'm3_summary.txt'), 'w') as f:
    f.write("N, alpha*  (interior min |x|<=0.95L crossing zero)\n")
    for N, astar in results:
        f.write(f"{N}\t{astar}\n")
    f.write(f"\nspread (max-min) = {spread:.6e}\n")
    f.write(f"mean = {alphas.mean():.6f}\n")
    f.write(f"relative spread = {rel_spread:.4%}\n")

print("\nwrote m3_summary.txt")
