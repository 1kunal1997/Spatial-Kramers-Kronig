"""
Route 2 (numeric, own construction): build SMOOTH, compactly-supported drop densities
g(t) >= 0 (not point masses -- a genuinely different, less degenerate test than route 1),
support(g) subset [-a,b], enforce the balance condition M1 = 0 EXACTLY via a closed-form
single compensator bump (derived below, not copied from any off-limits file -- in particular
NOT using theory/passivity_2026Aug/profile_generator.py, which has not been opened).

Then run the ACTUAL production DiHiTI code, tmm_helper.ht_derivative (explicitly whitelisted
shared infrastructure -- this is the object whose passivity verdict the claim is about, so
using it as ground truth is correct, not a violation of blindness), and check the sign of the
interior minimum of eps''(x) against the predicted a+b<=L threshold.

Profile families:
  - single Hann bump (smooth, compact support, C1 at edges)
  - flat-top plateau with Hann-tapered edges
  - multimodal: 2-3 Hann bumps of different heights/widths within [-a,b]
  - "near-atom": very narrow Hann bump (stress test approaching the route-1 point-mass limit)

M1 enforcement: g_total = g_main + A_comp * comp_shape, where comp_shape is a Hann bump placed
either near -a (if M1_main > 0) or near +b (if M1_main < 0), so its moment has the opposite
sign needed. Since M1 is LINEAR in g, A_comp = -M1_main / I_comp solves M1_total = 0 exactly
(up to quadrature error, which we make small with a fine sub-grid). A_comp comes out positive
by the sign-matched placement (tan(pi t/2L) < 0 for t<0, > 0 for t>0), so g_total stays >= 0.
"""
import numpy as np
import sys, os, json, datetime, subprocess
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT)
import tmm_helper as tmm_h

OUTDIR = os.path.dirname(os.path.abspath(__file__))
L = 1.0

# ---------------- shape primitives ----------------
def hann_bump(t, center, half_width, amp):
    out = np.zeros_like(t)
    mask = np.abs(t - center) < half_width
    out[mask] = amp * 0.5 * (1 + np.cos(np.pi * (t[mask] - center) / half_width))
    return out

def plateau(t, left, right, taper, amp):
    out = np.zeros_like(t)
    flat = (t >= left) & (t <= right)
    out[flat] = amp
    lt = (t >= left - taper) & (t < left)
    out[lt] = amp * 0.5 * (1 + np.cos(np.pi * (left - t[lt]) / taper))
    rt = (t > right) & (t <= right + taper)
    out[rt] = amp * 0.5 * (1 + np.cos(np.pi * (t[rt] - right) / taper))
    return out

def tan_weight(t, L):
    return np.tan(np.pi * t / (2 * L))

def moment_M1(t, g):
    return np.trapz(g * tan_weight(t, L), t)

def enforce_M1_zero(t, g_main, a, b, L, comp_width_frac=0.15):
    """Add one Hann compensator bump (placed near -a or +b, whichever sign is needed) so
    that the TOTAL g has M1 exactly 0 (up to quadrature error)."""
    M1_main = moment_M1(t, g_main)
    if abs(M1_main) < 1e-14:
        return g_main.copy(), 0.0, M1_main, "none"
    if M1_main > 0:
        # need a negative-moment contribution -> place near -a (t<0 side)
        cw = comp_width_frac * a
        cw = min(cw, 0.9 * a)
        center = -a + cw  # so support [center-cw, center+cw] = [-a, -a+2cw] subset [-a, b]
        comp_shape = hann_bump(t, center, cw, 1.0)
        side = "neg"
    else:
        cw = comp_width_frac * b
        cw = min(cw, 0.9 * b)
        center = b - cw
        comp_shape = hann_bump(t, center, cw, 1.0)
        side = "pos"
    I_comp = moment_M1(t, comp_shape)
    A_comp = -M1_main / I_comp
    g_total = g_main + A_comp * comp_shape
    M1_after = moment_M1(t, g_total)
    return g_total, A_comp, M1_after, side

def build_g(t, a, b, L, family, seed=0):
    rng = np.random.default_rng(seed)
    if family == "single_bump_centered":
        c = (b - a) / 2.0
        hw = min(a, b) * 0.85
        g = hann_bump(t, c, hw, 1.0)
    elif family == "single_bump_offcenter_left":
        c = -a * 0.6
        hw = a * 0.35
        g = hann_bump(t, c, hw, 1.0)
    elif family == "single_bump_offcenter_right":
        c = b * 0.6
        hw = b * 0.35
        g = hann_bump(t, c, hw, 1.0)
    elif family == "plateau_fill":
        taper = 0.15 * (a + b)
        left = -a + taper
        right = b - taper
        if right <= left:
            left, right = -a * 0.3, b * 0.3
        g = plateau(t, left, right, taper, 1.0)
    elif family == "near_atom_left":
        g = hann_bump(t, -a * 0.9, max(0.01, 0.03 * a), 1.0)
    elif family == "near_atom_right":
        g = hann_bump(t, b * 0.9, max(0.01, 0.03 * b), 1.0)
    elif family == "multimodal_3bump":
        centers = [-a * 0.7, (b - a) * 0.15, b * 0.7]
        widths = [0.15 * a, 0.12 * (a + b), 0.15 * b]
        amps = [0.7, 1.0, 0.5]
        g = np.zeros_like(t)
        for c, w, am in zip(centers, widths, amps):
            g += hann_bump(t, c, max(w, 1e-3), am)
    elif family == "multimodal_random":
        n_bumps = 4
        g = np.zeros_like(t)
        for _ in range(n_bumps):
            c = rng.uniform(-a * 0.85, b * 0.85)
            w = rng.uniform(0.03, 0.2) * (a + b)
            w = min(w, min(c + a, b - c) * 0.95) if (c + a) > 0 and (b - c) > 0 else 0.02*(a+b)
            w = max(w, 1e-3)
            am = rng.uniform(0.3, 1.0)
            g += hann_bump(t, c, w, am)
    else:
        raise ValueError(family)
    return g

# ---------------- pipeline ----------------
def run_case(a, b, family, N, seed=0, edge_margin=0.01):
    xx = np.linspace(-L, L, N)
    g_main = build_g(xx, a, b, L, family, seed=seed)
    g_total, A_comp, M1_after, side = enforce_M1_zero(xx, g_main, a, b, L)
    assert np.all(g_total >= -1e-12), f"g went negative! min={g_total.min()}"
    # eps'(x): monotone non-increasing, drop density g => deps'/dx = -g
    eps_re = -np.concatenate([[0.0], np.cumsum(0.5 * (g_total[1:] + g_total[:-1]) * np.diff(xx))])
    eps_pp = tmm_h.ht_derivative(xx, eps_re)
    # interior min: full domain minus a small margin at +/-L (structurally pinned zero)
    xL = (1 - edge_margin) * L
    mask = np.abs(xx) <= xL
    mn = eps_pp[mask].min()
    xmin = xx[mask][np.argmin(eps_pp[mask])]
    return dict(a=a, b=b, family=family, N=N, M1_after=float(M1_after), A_comp=float(A_comp),
                side=side, g_max=float(g_total.max()), interior_min_eps_pp=float(mn),
                argmin_x=float(xmin), eps_pp_endpoints=(float(eps_pp[0]), float(eps_pp[-1])))

# ---------------- sweep definitions ----------------
families = ["single_bump_centered", "single_bump_offcenter_left", "single_bump_offcenter_right",
            "plateau_fill", "near_atom_left", "near_atom_right", "multimodal_3bump", "multimodal_random"]

sweep_ab = {
    "symmetric": [(0.30, 0.30), (0.45, 0.45), (0.49, 0.49), (0.499, 0.499), (0.500, 0.500),
                  (0.501, 0.501), (0.51, 0.51), (0.55, 0.55), (0.70, 0.70)],
    "asym_L4_3L4": [(0.20, 0.70), (0.24, 0.75), (0.249, 0.750), (0.250, 0.750), (0.251, 0.751),
                    (0.26, 0.76), (0.30, 0.80)],
    "strong_asym": [(0.05, 0.90), (0.05, 0.94), (0.05, 0.949), (0.05, 0.950), (0.05, 0.951),
                    (0.05, 0.96), (0.05, 0.98)],
}

N_MAIN = 16001

results = []
for grp, pairs in sweep_ab.items():
    for a, b in pairs:
        for fam in families:
            try:
                r = run_case(a, b, fam, N_MAIN, seed=hash((grp, a, b, fam)) % (2**31))
            except Exception as e:
                print(f"SKIP {grp} a={a} b={b} {fam}: {e}")
                continue
            apb = a + b
            predicted_pass = apb <= L
            actual_pass = r["interior_min_eps_pp"] >= -1e-6  # numeric noise floor
            r.update(group=grp, a_plus_b_over_L=apb / L, predicted_pass=predicted_pass,
                      actual_pass=bool(actual_pass), agree=bool(predicted_pass == actual_pass))
            results.append(r)

n_total = len(results)
n_agree = sum(r["agree"] for r in results)
print(f"Route-2 numeric (N={N_MAIN}): {n_agree}/{n_total} agree with a+b<=L threshold")

mismatches = [r for r in results if not r["agree"]]
for r in mismatches:
    print("MISMATCH:", {k: r[k] for k in ("group","a","b","family","a_plus_b_over_L",
                                            "interior_min_eps_pp","argmin_x","predicted_pass","actual_pass")})

with open(os.path.join(OUTDIR, "route2_results.json"), "w") as f:
    json.dump(dict(N=N_MAIN, results=results, n_total=n_total, n_agree=n_agree,
                    timestamp=datetime.datetime.utcnow().isoformat()), f, indent=2, default=float)
print("wrote route2_results.json")
