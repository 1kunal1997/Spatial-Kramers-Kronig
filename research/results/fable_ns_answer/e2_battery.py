"""E2 — Theorem 2 witness: NO finite battery of linear functionals decides passivity.

Construction (matches the proof in ns_answer.md):
  n = k+2 balanced pairs with pairwise-distinct atoms (some super-threshold "bad",
  some sub-threshold "good"). In the mass orthant R^n_+, f(m) = sup_eta Delta_m(eta)
  is convex; walk to a boundary point p (f=0, unique finite active eta*). The battery
  matrix M (k x n) has kernel of dim >= 2; pick a kernel direction kappa with
  <grad f, kappa> != 0 (Danskin: grad f = (Delta_{P_j}(eta*))_j). If the kernel is
  orthogonal to the gradient at one boundary point, move ALONG the boundary (the
  proof's IFT step: eta* varies, and the vectors v(eta*) cannot stay inside the
  k-dim row space on an eta-interval, by the independence lemma). Then
      g_P = p - tau*kappa   (strictly passive)
      g_G = p + tau*kappa   (strict gain)
  have IDENTICAL battery readings and IDENTICAL atom sets (all weights positive).

Battery 1 (k=8) deliberately contains the ENTIRE known necessary cascade:
  mass, M1, M2, sigma  +  moments t, t^2, t^3, t^4.
Battery 2 (k=16) adds t^5..t^8, cos/sin(pi t), cos/sin(2 pi t).

Controls:
  (a) a NON-kernel direction changes the battery readings (matching is doing work);
  (b) verdicts from the exact node test are corroborated by production ht_derivative;
  (c) Lemma A sanity: the n functions Delta_j(eta) are numerically independent
      (smallest singular value of their sampled matrix is > 0);
  (d) kernel quality: max |M @ kappa| reported.
"""
import numpy as np
from common import (Atomic, balanced_pair, exact_sup_gap,
                    production_min_eps2, apply_battery)

rng = np.random.default_rng(7)

def make_battery(k):
    phis = [lambda t: np.ones_like(t),
            lambda t: np.tan(np.pi * t / 2),
            lambda t: 1 / np.cos(np.pi * t / 2) ** 2,
            lambda t: np.log(2 * np.cos(np.pi * t / 2)),
            lambda t: t, lambda t: t**2, lambda t: t**3, lambda t: t**4]
    if k == 16:
        phis += [lambda t: t**5, lambda t: t**6, lambda t: t**7, lambda t: t**8,
                 lambda t: np.cos(np.pi * t), lambda t: np.sin(np.pi * t),
                 lambda t: np.cos(2 * np.pi * t), lambda t: np.sin(2 * np.pi * t)]
    return phis[:k]

def run(k, bads, goods, tau_scale=0.8, tries=30, prod_Ns=(2**16,), prod_s=1.5e-3):
    pairs = [balanced_pair(a, b) for (a, b) in bads + goods]
    nb, n = len(bads), len(pairs)
    assert n == k + 2
    atoms = np.concatenate([p.t for p in pairs])
    assert len(np.unique(np.round(atoms, 12))) == 2 * n, "atom collision"

    def mix(masses):
        t = np.concatenate([p.t for p in pairs])
        w = np.concatenate([mi * p.w for mi, p in zip(masses, pairs)])
        return Atomic(t, w)

    phis = make_battery(k)
    Mmat = np.column_stack([apply_battery(phis, pj) for pj in pairs])  # k x n
    # kernel with a conservative rank cut (relative tolerance on singular values)
    U, sv, Vt = np.linalg.svd(Mmat)
    rank = int(np.sum(sv > 1e-10 * sv[0]))
    kern = Vt[rank:]
    print(f"  battery rank = {rank} (k = {k}), dim ker = {kern.shape[0]}, "
          f"sv range {sv[0]:.1e}..{sv[rank-1]:.1e}")

    def boundary_point(good_masses):
        def f_of(s):
            return exact_sup_gap(mix(np.concatenate([np.full(nb, s), good_masses])))[0]
        lo, hi = 0.0, 1.0
        while f_of(hi) < 0: hi *= 2
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            (lo, hi) = (mid, hi) if f_of(mid) < 0 else (lo, mid)
        return np.concatenate([np.full(nb, 0.5 * (lo + hi)), good_masses])

    # walk along the boundary until the kernel sees the gradient
    chosen = None
    for it in range(tries):
        gm = np.ones(n - nb) if it == 0 else rng.uniform(0.2, 3.0, n - nb)
        p = boundary_point(gm)
        sup_p, eta_star = exact_sup_gap(mix(p))
        grad = np.array([pj.Delta(np.array([eta_star]))[0] for pj in pairs])
        proj = kern @ grad
        score = np.linalg.norm(proj)
        if it == 0 or score > chosen[0]:
            chosen = (score, p, sup_p, eta_star, grad, proj)
        if score > 1e-3:
            break
    score, p, sup_p, eta_star, grad, proj = chosen
    print(f"  boundary point ({it+1} tried): sup Delta = {sup_p:+.2e} at "
          f"eta* = {eta_star:+.6f},  |P_ker grad| = {score:.4f}")
    assert score > 1e-6, "no boundary point with kernel-visible gradient found"

    kappa = kern.T @ proj
    kappa /= np.linalg.norm(kappa)
    print(f"  kernel quality: max |M kappa| = {np.abs(Mmat @ kappa).max():.2e}")

    tau = tau_scale * p.min() / max(1e-12, np.abs(kappa).max())
    sgn = np.sign(grad @ kappa)
    mG, mP = p + sgn * tau * kappa, p - sgn * tau * kappa
    assert mG.min() > 0 and mP.min() > 0
    gG, gP = mix(mG), mix(mP)

    lG, lP = apply_battery(phis, gG), apply_battery(phis, gP)
    rel = np.max(np.abs(lG - lP) / (1 + np.abs(lG)))
    supG, eG = exact_sup_gap(gG)
    supP, eP = exact_sup_gap(gP)
    print(f"  battery match:   max rel |l(gG) - l(gP)| = {rel:.2e}")
    print(f"  M1 exact:        {gG.M1():+.2e} / {gP.M1():+.2e}")
    print(f"  verdict gG:      sup Delta = {supG:+.6e} at eta = {eG:+.4f}  -> "
          f"{'GAIN' if supG > 0 else 'PASSIVE'}   min eps'' = {-supG/np.pi:+.3e}")
    print(f"  verdict gP:      sup Delta = {supP:+.6e} at eta = {eP:+.4f}  -> "
          f"{'GAIN' if supP > 0 else 'PASSIVE'}")
    print(f"  same atoms, min weight: gG {gG.w.min():.3e} / gP {gP.w.min():.3e}")

    d = rng.standard_normal(n); d /= np.linalg.norm(d)
    lD = apply_battery(phis, mix(p + 0.5 * tau * np.abs(d)))
    lp0 = apply_battery(phis, mix(p))
    print(f"  control non-kernel dir: max rel change = "
          f"{np.max(np.abs(lD - lp0)/(1+np.abs(lp0))):.2e}  (>> match above)")

    # production corroboration; near-threshold witnesses need large N because the
    # production pipeline carries a common-mode O(h) offset on staircase profiles
    # (documented in ns_answer.md; it cancels in the gG/gP difference)
    for N in prod_Ns:
        pvG, pxG = production_min_eps2(gG, N=N, s=prod_s)
        pvP, pxP = production_min_eps2(gP, N=N, s=prod_s)
        print(f"  production N=2^{int(np.log2(N))}: gG min = {pvG:+.3e} @ {pxG:+.4f} "
              f"({'GAIN' if pvG < -1e-9 else 'PASS'}), gP min = {pvP:+.3e} "
              f"({'GAIN' if pvP < -1e-9 else 'PASS'})")

    np.savez(f"witness_k{k}.npz", t=gG.t, wG=gG.w, wP=gP.w,
             readings_G=lG, readings_P=lP, eta_star=eta_star)

    eta_grid = np.tan(np.pi * 0.5 * np.linspace(-0.995, 0.995, 4001))
    G = np.column_stack([pj.Delta(eta_grid) for pj in pairs])
    svG = np.linalg.svd(G, compute_uv=False)
    print(f"  Lemma A sanity: sing. vals of [Delta_j(eta)] in [{svG[-1]:.3e}, {svG[0]:.3e}]")
    # rel is limited by the float SVD kernel of a nearly-degenerate battery
    # (exact-arithmetic matching is what the theorem proves); 1e-10 is the honest bar
    return supG > 0 and supP < 0 and rel < 1e-10

print("=" * 78)
print("Battery 1: k = 8  (mass, M1, M2, sigma, t..t^4)  --  n = 10 pairs")
bads8 = [(a, 1.15 - a) for a in (0.20, 0.25, 0.30, 0.35, 0.40)]
goods8 = [(a, 0.60 - a) for a in (0.10, 0.12, 0.14, 0.16, 0.18)]
ok1 = run(8, bads8, goods8, prod_Ns=(2**16,), prod_s=1.5e-3)

print("=" * 78)
print("Battery 2: k = 16 (adds t^5..t^8, cos/sin(pi t), cos/sin(2 pi t)) -- n = 18 pairs")
bads16 = [(round(a, 4), round(1.15 - a, 4)) for a in np.linspace(0.20, 0.44, 9)]
# good-pair positive atoms straddle the mixture's active point x* ~ 0.59, where the
# log kernel's singularity defeats smooth-battery approximation (see ns_answer.md)
goods16 = [(round(a, 4), round(0.75 - a, 4)) for a in np.linspace(0.095, 0.185, 9)]
ok2 = run(16, bads16, goods16, prod_Ns=(2**20, 2**22), prod_s=6e-4)

print("=" * 78)
print(f"THEOREM 2 WITNESSES: battery-8 {'OK' if ok1 else 'FAIL'}, "
      f"battery-16 {'OK' if ok2 else 'FAIL'}")
