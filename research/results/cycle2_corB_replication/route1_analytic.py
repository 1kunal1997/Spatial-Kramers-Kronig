"""
Route 1 (analytic, strongest): closed-form eps''(x) for a TWO-ATOM drop density
g(t) = A0*delta(t-t0) + A1*delta(t-t1), t0 in (-L,0), t1 in (0,L), derived from
scratch via Fourier series on the circle -- NOT copied from any file in
theory/passivity_2026Aug/ or the h9_* files (none of which have been opened).

DERIVATION (self-contained, done here, not read from any off-limits source):

Domain period P = 2L. For period-P functions, Fourier modes e^{i n pi x / L}.
Hilbert transform (circular/periodic) acts on modes as H[e^{inphi}] = -i sgn(n) e^{inphi}.

Periodic delta comb: delta_P(x-t0) = (1/P) sum_n e^{i n pi (x-t0)/L}.

H[delta_P(.-t0)](x) = (1/P) sum_{n != 0} (-i sgn n) e^{i n pi (x-t0)/L}
                     = (2/P) sum_{n=1}^inf sin(n pi (x-t0)/L)
Using the classical Fourier-series identity (Abel/PV summed):
    sum_{n=1}^inf sin(n*phi) = (1/2) cot(phi/2),   phi = pi(x-t0)/L
=>  H[delta_P(.-t0)](x) = (1/P) cot( pi(x-t0)/(2L) ) = (1/(2L)) cot( pi(x-t0)/(2L) )

DiHiTI pipeline: u = deps'/dx = -g(x)  (since g = -deps'/dt)
                 v = H[u]
                 eps'' = antiderivative(v) - linear detilt

For g = A0 delta(t-t0) + A1 delta(t-t1):
  u = -A0 delta(x-t0) - A1 delta(x-t1)
  v(x) = -(A0/(2L)) cot(pi(x-t0)/(2L)) - (A1/(2L)) cot(pi(x-t1)/(2L))

Antiderivative: INT cot(pi(s-t0)/(2L)) ds = (2L/pi) ln|sin(pi(s-t0)/(2L))| + C
  raw(x) = -(A0/pi) ln|sin(pi(x-t0)/(2L))| - (A1/pi) ln|sin(pi(x-t1)/(2L))| + C

Fix C by raw(-L) = 0 (matches cumulative_trapezoid(..., initial=0) convention):
  sin(pi(-L-t0)/(2L)) = -cos(pi t0/(2L))  =>  |...| = cos(pi t0/(2L))  (t0 in (-L,L))
  => C = (A0/pi) ln[cos(pi t0/(2L))] + (A1/pi) ln[cos(pi t1/(2L))]

  eps''_raw(x) = (A0/pi) ln[ cos(pi t0/(2L)) / |sin(pi(x-t0)/(2L))| ]
               + (A1/pi) ln[ cos(pi t1/(2L)) / |sin(pi(x-t1)/(2L))| ]

Check at x=+L: sin(pi(L-t0)/(2L)) = cos(pi t0/(2L))  =>  A0-term = ln(1) = 0. Same for A1.
So eps''_raw(+L) = 0 automatically -- the straight-line detilt is the zero line, no-op
for this construction. (This is itself a nontrivial, independently-derived check on V5:
endpoints pinned to zero even before detilt, for point-mass drop densities.)

BALANCE CONDITION M1=0:
  M1 = A0 tan(pi t0/(2L)) + A1 tan(pi t1/(2L)) = 0
  => A1 = -A0 * tan(pi t0/(2L)) / tan(pi t1/(2L))
  For t0 in (-L,0), t1 in (0,L): tan(pi t0/2L) < 0, tan(pi t1/2L) > 0 => A1/A0 > 0. Always solvable.

SIGN CALIBRATION: scipy's hilbert() analytic-signal convention could differ from the
"H[e^{i n phi}] = -i sgn(n) e^{i n phi}" convention by an overall sign. We calibrate this
numerically against a simple test signal (a narrow bump standing in for a delta) BEFORE
using it to interpret any pass/fail result, and report the calibration explicitly.
"""
import numpy as np
from scipy.signal import hilbert as scipy_hilbert
import json, datetime, subprocess, os

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Step 0: sign calibration against scipy.signal.hilbert ----------
def calibrate_sign(L=1.0, N=20001):
    x = np.linspace(-L, L, N)
    t0 = 0.0
    w = 0.01 * L  # narrow bump standing in for a delta
    # normalized narrow Gaussian-like bump (smooth, not truly periodic delta, just for sign calibration)
    bump = np.exp(-0.5 * ((x - t0) / w) ** 2)
    bump /= np.trapz(bump, x)  # unit mass, approx delta
    v_scipy = np.imag(scipy_hilbert(bump))
    # my closed form for H[delta_P(.-t0)](x) = (1/(2L)) cot(pi(x-t0)/(2L)), evaluated away from t0
    mask = np.abs(x - t0) > 5 * w
    v_mine = (1.0 / (2 * L)) / np.tan(np.pi * (x - t0) / (2 * L))
    # compare sign/scale on points away from the singularity
    ratio = v_scipy[mask] / v_mine[mask]
    return float(np.median(ratio)), x, v_scipy, v_mine, mask

sign_ratio, x_cal, v_scipy_cal, v_mine_cal, mask_cal = calibrate_sign()
print(f"[calibration] median(scipy_v / analytic_v) on far-field points = {sign_ratio:.6f}")
SIGN = np.sign(sign_ratio)
SCALE_CHECK = abs(abs(sign_ratio) - 1.0)
print(f"[calibration] |ratio|-1 = {SCALE_CHECK:.4e} (should be small -- confirms scale/kernel form, not just sign)")
print(f"[calibration] adopted SIGN = {SIGN:+.0f}")

# ---------- Step 1: closed form eps''(x) for the two-atom construction ----------
def eps_pp_two_atom(x, L, t0, t1, A0, A1, sign=SIGN):
    """Closed-form eps''(x), derived above, with calibrated sign applied to v (hence to the result)."""
    term0 = (A0 / np.pi) * np.log(np.cos(np.pi * t0 / (2 * L)) / np.abs(np.sin(np.pi * (x - t0) / (2 * L))))
    term1 = (A1 / np.pi) * np.log(np.cos(np.pi * t1 / (2 * L)) / np.abs(np.sin(np.pi * (x - t1) / (2 * L))))
    return sign * (term0 + term1)

def solve_A1(L, t0, t1, A0=1.0):
    return -A0 * np.tan(np.pi * t0 / (2 * L)) / np.tan(np.pi * t1 / (2 * L))

def interior_min(L, t0, t1, A0, A1, N=400001, sign=SIGN, edge_margin=1e-4):
    # Search the FULL domain except: (i) a tiny margin at the structurally-pinned
    # endpoints +/-L (edge_margin*L), and (ii) tiny neighborhoods of the two point-mass
    # singularities t0,t1 where eps'' -> +inf (not a candidate minimum, exclude to avoid
    # numerical overflow, not to bias the search).
    xL = (1 - edge_margin) * L
    x = np.linspace(-xL, xL, N)
    excl = 1e-6 * L
    x = x[(np.abs(x - t0) > excl) & (np.abs(x - t1) > excl)]
    vals = eps_pp_two_atom(x, L, t0, t1, A0, A1, sign=sign)
    idx = np.argmin(vals)
    return vals[idx], x[idx]

# ---------- Step 2: sweep a+b vs L, symmetric and strongly asymmetric ----------
L = 1.0
cases = []
# (label, a, b)  -- atoms placed at t0=-a, t1=+b (boundary of support, hardest case)
sweep_defs = [
    ("symmetric", [(0.30, 0.30), (0.45, 0.45), (0.49, 0.49), (0.499,0.499), (0.50, 0.50), (0.501,0.501), (0.51, 0.51), (0.55, 0.55), (0.70, 0.70)]),
    ("asymmetric_L4_3L4", [(0.20, 0.70), (0.23,0.75), (0.24,0.759), (0.25, 0.75), (0.251,0.751), (0.26, 0.76), (0.30, 0.80)]),
    ("strongly_asymmetric", [(0.05, 0.90), (0.05, 0.94), (0.05, 0.949), (0.05, 0.95), (0.05, 0.951), (0.05, 0.96), (0.05, 0.99)]),
    ("near_full_one_side", [(0.02, 0.95), (0.02, 0.979), (0.02, 0.98), (0.02, 0.981), (0.02, 0.99)]),
]

results = {}
for label, pairs in sweep_defs:
    results[label] = []
    for a, b in pairs:
        t0, t1 = -a, b
        A0 = 1.0
        A1 = solve_A1(L, t0, t1, A0)
        # verify M1=0
        M1 = A0*np.tan(np.pi*t0/(2*L)) + A1*np.tan(np.pi*t1/(2*L))
        mn, xmin = interior_min(L, t0, t1, A0, A1)
        apb = a + b
        predicted_pass = apb <= L
        actual_pass = mn >= -1e-9  # closed form is essentially exact; tiny numerical epsilon
        results[label].append(dict(a=a, b=b, a_plus_b=apb, a_plus_b_over_L=apb/L,
                                    A0=A0, A1=A1, M1_check=M1,
                                    interior_min_eps_pp=mn, argmin_x=xmin,
                                    predicted_pass=predicted_pass, actual_pass=bool(actual_pass),
                                    agree=bool(predicted_pass == actual_pass)))
        print(f"[{label}] a={a:.3f} b={b:.3f} (a+b)/L={apb/L:.4f} M1={M1:.2e} "
              f"min eps''={mn:+.6e} @x={xmin:+.4f} predicted={'PASS' if predicted_pass else 'FAIL'} "
              f"actual={'PASS' if actual_pass else 'FAIL'} "
              f"{'OK' if predicted_pass==actual_pass else '*** MISMATCH ***'}")

all_agree = all(r['agree'] for label in results for r in results[label])
print(f"\nRoute-1 (analytic two-atom) overall agreement with (a+b<=L) threshold: {all_agree}")

manifest = dict(
    route="1_analytic_two_atom",
    timestamp=datetime.datetime.utcnow().isoformat(),
    git_sha=subprocess.run(["git","rev-parse","HEAD"],cwd=OUTDIR,capture_output=True,text=True).stdout.strip(),
    sign_calibration=dict(median_ratio=sign_ratio, adopted_sign=float(SIGN), scale_check_abs_ratio_minus_1=SCALE_CHECK),
    L=L,
    results=results,
    all_agree=all_agree,
)
with open(os.path.join(OUTDIR, "route1_results.json"), "w") as f:
    json.dump(manifest, f, indent=2, default=float)
print("wrote route1_results.json")
