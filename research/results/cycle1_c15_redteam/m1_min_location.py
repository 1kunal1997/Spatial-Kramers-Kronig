"""M1: where is the min of eps''? Full grid vs interior |x|<=0.95L, for >=200
profiles from profile_generator.sample(), plus logistic w=0.05 and an explicit
batch of opposite_bumps draws.

Grid convention: we use the generator's NATIVE L=2.5 (not rescaled to 1.0) at
N=2**14, because several families (triangle, cosine_bump, gaussian, tanh_slopes,
and the bump separations/widths in logistic_bump / opposite_bumps) are
calibrated in absolute x-units tied to L=2.5. Rescaling the domain to L=1.0
while keeping those absolute widths would silently change the physics (bumps
would become comparable to the whole domain). We are consistent throughout at
L=2.5, N=16384, and report normalized quantities (x/L, margin/D) so the choice
of L does not bias interpretation.
"""
import numpy as np
import csv
from common import pg, tmm_h, grid, dihiti, OUTDIR
import os

L = 2.5
N = 2**14

rng = np.random.default_rng(12345)


def record(name, params, x, e_re, e_im):
    L_ = x[-1]
    D = e_re[0] - e_re[-1]
    i_full = int(np.argmin(e_im))
    min_full = float(e_im[i_full])
    x_full = float(x[i_full] / L_)

    interior_mask = np.abs(x) <= 0.95 * L_
    e_im_int = e_im[interior_mask]
    x_int = x[interior_mask]
    i_int = int(np.argmin(e_im_int))
    min_int = float(e_im_int[i_int])
    x_int_loc = float(x_int[i_int] / L_)

    passive = min_full >= -1e-3 * max(float(e_im.max()), 1e-12)

    return dict(name=name, params=str(params), D=D,
                min_full=min_full, x_argmin_full=x_full,
                min_interior=min_int, x_argmin_interior=x_int_loc,
                min_interior_over_D=min_int / D if D != 0 else np.nan,
                passive=passive,
                endpoint_adjacent=bool(abs(x_full) > 0.99))


rows = []

# 200+ random draws across all families, at our fixed (L, N)
for s in pg.sample(220, rng=rng):
    name = s['name']
    params = s['params']
    x = grid(L, N)
    from profile_generator import PROFILES
    fn, defaults = PROFILES[name]
    kw = dict(defaults); kw.update(params)
    e_re = np.asarray(fn(x, **kw), float)
    e_im = dihiti(e_re, x)
    rows.append(record(name, params, x, e_re, e_im))

# explicit logistic w=0.05  (w := 1/k transition-width convention used in this project)
x = grid(L, N)
k = 1.0 / 0.05
e_re = tmm_h.logistic(x, k, pg.NB)
e_im = dihiti(e_re, x)
rows.append(record('logistic_w0.05', dict(k=k), x, e_re, e_im))

# explicit batch of opposite_bumps draws (beyond whatever sample() picked)
from profile_generator import PROFILES
fn, defaults = PROFILES['opposite_bumps']
for alpha in np.linspace(0.0, 2.5, 15):
    for sep, w in [(0.4, 0.25), (0.9, 0.45), (1.8, 0.8)]:
        kw = dict(defaults); kw.update(alpha=alpha, sep=sep, w=w)
        x = grid(L, N)
        e_re = np.asarray(fn(x, **kw), float)
        e_im = dihiti(e_re, x)
        rows.append(record('opposite_bumps_explicit', kw, x, e_re, e_im))

fieldnames = ['name', 'params', 'D', 'min_full', 'x_argmin_full', 'min_interior',
              'x_argmin_interior', 'min_interior_over_D', 'passive', 'endpoint_adjacent']
with open(os.path.join(OUTDIR, 'm1_raw.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

passive = [r for r in rows if r['passive']]
gain = [r for r in rows if not r['passive']]

print(f"total profiles: {len(rows)}   passive: {len(passive)}   gain: {len(gain)}")
frac_endpoint = np.mean([r['endpoint_adjacent'] for r in passive])
print(f"fraction of PASSIVE profiles with global min endpoint-adjacent (|x|/L>0.99): {frac_endpoint:.4f}")

# |D| ~ 0 profiles (e.g. cosine_bump: same eps' at both endpoints) blow up the
# min/D ratio without being a real "near-zero-margin" story. Report them
# separately and exclude from the percentile/histogram of the ratio.
D_vals = np.array([r['D'] for r in passive], dtype=float)
Dscale = np.median(np.abs(D_vals[np.abs(D_vals) > 1e-9])) if np.any(np.abs(D_vals) > 1e-9) else 1.0
smallD = [r for r in passive if abs(r['D']) < 1e-3 * Dscale]
normalD = [r for r in passive if abs(r['D']) >= 1e-3 * Dscale]
print(f"\npassive profiles with |D| ~ 0 (excluded from ratio stats): {len(smallD)} / {len(passive)}")
if smallD:
    names_smallD = sorted(set(r['name'] for r in smallD))
    print(f"  families involved: {names_smallD}")

vals = np.array([r['min_interior_over_D'] for r in normalD], dtype=float)
vals = vals[np.isfinite(vals)]
pct = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99])
print("percentiles of interior_min/D among PASSIVE profiles:")
for p, v in zip([1, 5, 25, 50, 75, 95, 99], pct):
    print(f"  {p:>3}th pct: {v:.6e}")

# histogram (text form, since this is a script not a notebook)
hist, edges = np.histogram(vals, bins=30)
print("\nhistogram of interior_min/D (passive):")
for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
    print(f"  [{lo:+.4e}, {hi:+.4e}): {'#'*int(h)} ({h})")

with open(os.path.join(OUTDIR, 'm1_summary.txt'), 'w') as f:
    f.write(f"total profiles: {len(rows)}   passive: {len(passive)}   gain: {len(gain)}\n")
    f.write(f"fraction of PASSIVE profiles with global min endpoint-adjacent (|x|/L>0.99): {frac_endpoint:.4f}\n")
    f.write(f"passive profiles with |D|~0 excluded from ratio stats: {len(smallD)} / {len(passive)}\n")
    if smallD:
        f.write(f"  families involved: {sorted(set(r['name'] for r in smallD))}\n")
    f.write("percentiles of interior_min/D among PASSIVE profiles (|D| not ~0):\n")
    for p, v in zip([1, 5, 25, 50, 75, 95, 99], pct):
        f.write(f"  {p:>3}th pct: {v:.6e}\n")
    f.write("\nhistogram of interior_min/D (passive), 30 bins:\n")
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        f.write(f"  [{lo:+.4e}, {hi:+.4e}): {h}\n")

print("\nwrote m1_raw.csv and m1_summary.txt")
