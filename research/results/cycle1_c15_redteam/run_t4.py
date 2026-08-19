import numpy as np
import scipy, timeit, json
from bench import onehilb, dihiti, profiles_at_N, sign_min_interior, L, A

out = []
def w(s=""):
    print(s)
    out.append(s)

w("## T4 candidate C: adaptive coarse-to-fine minimum search\n")
w("Procedure: evaluate eps'' via onehilb on a coarse grid (n0 points), find the coarse-grid argmin,\n")
w("then refine by evaluating onehilb on a small local window around that argmin at full resolution\n")
w("(a local dense re-sample + onehilb applied only to the windowed region, still needs a Hilbert transform\n")
w("over some domain since HT is nonlocal -- we approximate by recomputing full onehilb at a modestly larger n\n")
w("and repeating, i.e. simple coarse-to-fine doubling until the reported minimum stabilizes to 1% relative.\n")

N_GT = 2**16
x_gt, profs_gt = profiles_at_N(N_GT)

gt = {}
for name, u in profs_gt.items():
    v = onehilb(u, x_gt)
    s, m = sign_min_interior(v, x_gt)
    gt[name] = (s, m)

def adaptive_search(u_full, x_full, n0=32, tol=0.01, nmax=None):
    """Coarse-to-fine: double n until interior-min value stabilizes to `tol` relative
    between successive levels. Returns (final_n, sign, min_value, n_evals_total)."""
    N = len(x_full)
    if nmax is None:
        nmax = N
    n = n0
    prev_m = None
    total_evals = 0
    while True:
        step = max(1, N // n)
        idx = np.arange(0, N, step)
        xu = x_full[idx]; uu = u_full[idx]
        v = onehilb(uu, xu)
        total_evals += len(idx)
        s, m = sign_min_interior(v, xu)
        if prev_m is not None:
            denom = max(abs(prev_m), 1e-12)
            if abs(m - prev_m) / denom < tol:
                return n, s, m, total_evals
        prev_m = m
        n *= 2
        if n > nmax:
            return n // 2, s, m, total_evals

w("\n### Evaluations needed vs N, and verdict correctness\n")
w("| profile | N | adaptive final n | total eps'' evals (sum over levels) | verdict matches GT? |")
w("|---|---|---|---|---|")
n_evals_list = []
mismatches = []
for name, u in profs_gt.items():
    s_gt, m_gt = gt[name]
    n_final, s, m, evals = adaptive_search(u, x_gt, n0=32, tol=0.01, nmax=N_GT)
    match = (s == s_gt)
    if not match:
        mismatches.append(name)
    n_evals_list.append(evals)
    w(f"| {name} | {N_GT} | {n_final} | {evals} | {match} |")

w(f"\nMean total evals: {np.mean(n_evals_list):.0f}  (vs N={N_GT}, ratio {np.mean(n_evals_list)/N_GT:.4f})")
w(f"Verdict mismatches vs GT: {mismatches if mismatches else 'NONE'}")

# ---------------- wall-clock comparison at N=2^14 ----------------
w("\n### Wall-clock: adaptive search vs plain FFT baseline, at N=2^14\n")
N = 2**14
x14, profs14 = profiles_at_N(N)
u_test = profs14["P1_logistic_w0.05"]

def run_baseline():
    return dihiti(u_test, x14).min()

def run_adaptive():
    return adaptive_search(u_test, x14, n0=32, tol=0.01, nmax=N)

t_base = timeit.repeat(run_baseline, repeat=20, number=1)
t_adapt = timeit.repeat(run_adaptive, repeat=20, number=1)
best_base = np.min(t_base)*1e6
best_adapt = np.min(t_adapt)*1e6
w(f"baseline dihiti:    best {best_base:.1f} us")
w(f"adaptive search:     best {best_adapt:.1f} us")
w(f"speedup: {best_base/best_adapt:.2f}x")

# also time a single full-N onehilb call as reference for "one-shot FFT" cost
def run_oneshot():
    return onehilb(u_test, x14).min()
t_one = timeit.repeat(run_oneshot, repeat=20, number=1)
best_one = np.min(t_one)*1e6
w(f"one-shot onehilb (single FFT, full N, for comparison): best {best_one:.1f} us")
w(f"adaptive vs one-shot onehilb speedup: {best_one/best_adapt:.2f}x")

json.dump({"mismatches": mismatches, "mean_evals": float(np.mean(n_evals_list)),
           "timing": {"baseline": best_base, "adaptive": best_adapt, "oneshot": best_one}},
          open("t4_results.json","w"), indent=2)

with open("t4_output.txt","w") as f:
    f.write("\n".join(out))
print("DONE T4")
