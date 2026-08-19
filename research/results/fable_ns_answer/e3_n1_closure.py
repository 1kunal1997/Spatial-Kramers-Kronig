"""E3 — closing board lead N1 ("necessity is about corner mass").

N1's true content: for each FAMILY (corner geometry + bulk geometry fixed), the
maximal passive corner-mass fraction lambda* is an exact, computable threshold —
but lambda* depends on the SHAPE AND POSITION OF THE BULK, ranging over an order
of magnitude for one and the same corner geometry. So no corner-mass functional
with any universal threshold can be two-sided. (Formally: corner mass is a linear
functional of g, so Theorem 2 already forbids it; this experiment quantifies the
failure.)

  (a) lambda* controls against red-team's independent bisection route:
      corner (-0.40,+0.90) bulk +-0.1 -> 0.262 [rt 0.262]; bulk +-0.2 -> 0.278
      [handA ~0.278]; symmetric corners +-0.90 -> 0.0872 [rt 0.087 @ 1.80L];
      +-0.99 -> 0.0009 [rt 0.0009 @ 1.98L].
      (Red-team's 1.40L and 1.94L rows used unrecorded corner splits — flagged.)
  (b) lambda* vs bulk, corner FIXED at (-0.4, +0.9): centred bulks give
      0.258..0.383; near-threshold bulks give down to 0.030. Order of magnitude.
  (c) killer pair: IDENTICAL corner distribution (corner pair (-0.4,+0.9),
      fraction 0.25): bulk pair (+-0.20) -> PASSIVE; bulk pair (-0.24,+0.74),
      also g >= 0, M1 = 0, sum 0.98 < L -> GAIN.
      No rule reading only g near the window ends can decide passivity.
"""
import numpy as np
from common import balanced_pair, exact_sup_gap, production_min_eps2

ETA = np.tan(np.pi * 0.5 * np.linspace(-0.999999, 0.999999, 400001))

def lam_star(corner, bulk):
    Dc, Db = corner.Delta(ETA), bulk.Delta(ETA)
    mask = Dc > 0
    return float(np.min((-Db[mask]) / (Dc[mask] - Db[mask])))

print("(a) lambda* controls vs red-team/handA")
corner = balanced_pair(0.4, 0.9)
print(f"  corner (-0.40,+0.90), bulk +-0.1: {lam_star(corner, balanced_pair(0.1,0.1)):.4f}  [rt 0.262]")
print(f"  corner (-0.40,+0.90), bulk +-0.2: {lam_star(corner, balanced_pair(0.2,0.2)):.4f}  [handA ~0.278]")
print(f"  corner +-0.90 (box 1.80), bulk +-0.1: {lam_star(balanced_pair(0.9,0.9), balanced_pair(0.1,0.1)):.4f}  [rt 0.087]")
print(f"  corner +-0.99 (box 1.98), bulk +-0.1: {lam_star(balanced_pair(0.99,0.99), balanced_pair(0.1,0.1)):.4f}  [rt 0.0009]")

print("\n(b) lambda* vs bulk, corner FIXED at (-0.4, +0.9)")
print("  centred bulk pairs +-m:")
for m in (0.05, 0.15, 0.25, 0.35, 0.45):
    print(f"    m={m:.2f}: lambda* = {lam_star(corner, balanced_pair(m, m)):.4f}")
print("  near-threshold bulk pairs (-a,+b), a+b -> L:")
for (a, b) in [(0.2, 0.7), (0.25, 0.7), (0.24, 0.74), (0.2, 0.79)]:
    print(f"    (-{a},+{b}) sum {a+b:.2f}: lambda* = {lam_star(corner, balanced_pair(a, b)):.4f}")

print("\n(c) killer pair: same corner distribution (fraction 0.25), different bulk")
for (a, b), expect in (((0.20, 0.20), "PASSIVE"), ((0.24, 0.74), "GAIN")):
    g = balanced_pair(0.4, 0.9, 0.25).mix(balanced_pair(a, b, 0.75))
    sup, eta = exact_sup_gap(g)
    verdict = "GAIN" if sup > 0 else "PASSIVE"
    pv, px = production_min_eps2(g, N=2**16, s=1e-3)
    print(f"  bulk (-{a},+{b}): exact sup Delta = {sup:+.4e} at eta={eta:+.3f} -> "
          f"{verdict} (expect {expect}); production min = {pv:+.3e} @ x={px:+.3f}")
