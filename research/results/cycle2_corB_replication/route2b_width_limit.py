"""
Route 2b: width-convergence study, triggered by an apparent Route-2 discrepancy.

FINDING THAT MOTIVATED THIS: many "single_bump_*", "plateau_fill", "multimodal_*" profiles
with a+b > L came out PASS (interior min eps'' > 0), contradicting the naive expectation that
a+b > L should always FAIL. Investigation showed the cause: those profiles' true numerical
"reach" (how close the bulk of g actually gets to -a and +b) was well short of the full box,
so their EFFECTIVE support was narrower than the nominal [-a,b] -- an under-adversarial
construction artifact, not a theorem violation. The "only if" direction of an iff bound like
this is a statement about SHARPNESS: there exists a profile (support touching -a and b) for
which a+b > L forces failure -- route 1's point-mass (delta) construction is exactly that
extremal profile, and it matched exactly (29/29, transition at a+b/L=1.0000 to machine
precision). This script checks whether SMOOTH profiles converge to that same sharp threshold
as their support is concentrated toward the true corners -a, +b (bump half-width w -> 0),
i.e. whether the delta-limit answer is the w->0 limit of a continuous family, as it must be
for the theorem to be internally consistent -- and quantifies how much "slack" (positive
margin despite a+b>L) exists at finite w.

Uses tmm_helper.ht_derivative (production DiHiTI code) as ground truth throughout.
"""
import numpy as np
import sys, os, json, datetime
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project")
import tmm_helper as tmm_h

OUTDIR = os.path.dirname(os.path.abspath(__file__))
L = 1.0

def hann_bump(t, center, half_width, amp):
    out = np.zeros_like(t)
    mask = np.abs(t - center) < half_width
    out[mask] = amp * 0.5 * (1 + np.cos(np.pi * (t[mask] - center) / half_width))
    return out

def tan_weight(t, L):
    return np.tan(np.pi * t / (2 * L))

def two_bump_case(xx, a, b, w_frac, L=1.0):
    """Two narrow Hann bumps: one near -a, one near +b, half-width = w_frac*(a+b).
    Amplitude ratio solved in closed form so M1 = 0 exactly (up to quadrature)."""
    w = w_frac * (a + b)
    w = min(w, 0.45 * a, 0.45 * b)  # keep bumps from overlapping / exceeding their side
    t0 = -a + w  # so support of bump0 touches -a exactly
    t1 = b - w   # support of bump1 touches b exactly
    s0 = hann_bump(xx, t0, w, 1.0)
    s1 = hann_bump(xx, t1, w, 1.0)
    I0 = np.trapz(s0 * tan_weight(xx, L), xx)
    I1 = np.trapz(s1 * tan_weight(xx, L), xx)
    # A0*I0 + A1*I1 = 0, A0=1 => A1 = -I0/I1
    A1 = -I0 / I1
    g = s0 + A1 * s1
    return g, w, t0, t1, A1

def interior_min_full(xx, eps_pp, edge_margin=0.01, L=1.0):
    xL = (1 - edge_margin) * L
    mask = np.abs(xx) <= xL
    mn = eps_pp[mask].min()
    xm = xx[mask][np.argmin(eps_pp[mask])]
    return mn, xm

def run(a, b, w_frac, N=32001):
    xx = np.linspace(-L, L, N)
    g, w, t0, t1, A1 = two_bump_case(xx, a, b, w_frac, L)
    assert (g >= -1e-10).all(), f"g negative, min={g.min()}"
    eps_re = -np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(xx))])
    eps_pp = tmm_h.ht_derivative(xx, eps_re)
    mn, xm = interior_min_full(xx, eps_pp)
    return dict(a=a, b=b, w_frac=w_frac, w=w, t0=t0, t1=t1, A1=A1, N=N,
                interior_min_eps_pp=float(mn), argmin_x=float(xm))

if __name__ == "__main__":
    print("="*100)
    print("Width-convergence study: does the smooth two-bump family approach the route-1")
    print("point-mass (delta) result as w -> 0, for a fixed a,b with a+b > L?")
    print("="*100)

    results = {}
    for (a, b) in [(0.55, 0.55), (0.25, 0.75), (0.05, 0.951)]:
        apb = a + b
        results[f"a={a}_b={b}"] = []
        print(f"\n--- a={a}, b={b}, (a+b)/L = {apb/L:.4f}  (predicted FAIL if a+b>L) ---")
        for w_frac in [0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
            r = run(a, b, w_frac, N=32001)
            results[f"a={a}_b={b}"].append(r)
            print(f"  w_frac={w_frac:<6} w={r['w']:.4f}  t0={r['t0']:+.4f} t1={r['t1']:+.4f} "
                  f"A1={r['A1']:.4f}  min eps''={r['interior_min_eps_pp']:+.6e} @x={r['argmin_x']:+.4f}")

    with open(os.path.join(OUTDIR, "route2b_results.json"), "w") as f:
        json.dump(dict(results=results, timestamp=datetime.datetime.utcnow().isoformat()),
                   f, indent=2, default=float)
    print("\nwrote route2b_results.json")
