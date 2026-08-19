import numpy as np
import scipy, timeit, json
from bench import (dihiti, onehilb, onehilb_rfft, profiles_at_N, sign_min_interior,
                    logistic_u, p2_u, p3_u, build_p4, A, L, nb)

out = []
def w(s=""):
    print(s)
    out.append(s)

w(f"scipy version: {scipy.__version__}, numpy version: {np.__version__}")

# ---------------- T1: baseline timing ----------------
w("\n## T1 baseline timing (dihiti + .min())\n")
w("| N | best (us) | median (us) |")
w("|---|---|---|")
t1_results = {}
for k in [12,13,14,15,16]:
    N = 2**k
    x = np.linspace(-L, L, N, endpoint=True)
    u = logistic_u(x, 0.05)
    def run():
        e = dihiti(u, x)
        return e.min()
    times = timeit.repeat(run, repeat=20, number=1)
    times_us = np.array(times)*1e6
    best = np.min(times_us)
    med = np.median(times_us)
    t1_results[N] = (best, med)
    w(f"| {N} | {best:.1f} | {med:.1f} |")

# ---------------- T2: onehilb agreement + timing ----------------
w("\n## T2 candidate A: single Hilbert transform\n")
N = 2**14
x, profs = profiles_at_N(N)

w("### (a) agreement dihiti vs onehilb at N=2^14\n")
w("| profile | rel rms (dihiti-onehilb)/max|dihiti| | max abs diff (|x|<0.99L) |")
w("|---|---|---|")
label_mismatch_full = []
label_mismatch_interior = []
rel_rms_all = {}
for name, u in profs.items():
    d = dihiti(u, x)
    o = onehilb(u, x)
    rel_rms = np.sqrt(np.mean(((d-o)/np.max(np.abs(d)))**2))
    mask = np.abs(x) < 0.99*L
    maxabs = np.max(np.abs(d[mask]-o[mask]))
    rel_rms_all[name] = rel_rms
    w(f"| {name} | {rel_rms:.3e} | {maxabs:.3e} |")
    # (b) label check
    sfull_d = np.sign(d.min()); sfull_o = np.sign(o.min())
    sint_d, _ = sign_min_interior(d, x); sint_o, _ = sign_min_interior(o, x)
    if sfull_d != sfull_o:
        label_mismatch_full.append(name)
    if sint_d != sint_o:
        label_mismatch_interior.append(name)

w("\n### (b) label (sign of min) disagreements dihiti vs onehilb\n")
w(f"Full-domain min sign mismatches: {label_mismatch_full if label_mismatch_full else 'NONE'}")
w(f"Interior (|x|<0.99L) min sign mismatches: {label_mismatch_interior if label_mismatch_interior else 'NONE'}")

w("\n### (c) timing at N=2^14\n")
u_test = profs["P1_logistic_w0.05"]
def run_dihiti():
    return dihiti(u_test, x).min()
def run_onehilb():
    return onehilb(u_test, x).min()
def run_onehilb_rfft():
    return onehilb_rfft(u_test, x).min()

t_dihiti = timeit.repeat(run_dihiti, repeat=20, number=1)
t_onehilb = timeit.repeat(run_onehilb, repeat=20, number=1)
t_rfft = timeit.repeat(run_onehilb_rfft, repeat=20, number=1)

best_d, med_d = np.min(t_dihiti)*1e6, np.median(t_dihiti)*1e6
best_o, med_o = np.min(t_onehilb)*1e6, np.median(t_onehilb)*1e6
best_r, med_r = np.min(t_rfft)*1e6, np.median(t_rfft)*1e6

w(f"dihiti:        best {best_d:.1f} us, median {med_d:.1f} us")
w(f"onehilb:       best {best_o:.1f} us, median {med_o:.1f} us  (speedup vs dihiti: {best_d/best_o:.2f}x)")
w(f"onehilb_rfft:  best {best_r:.1f} us, median {med_r:.1f} us  (speedup vs dihiti: {best_d/best_r:.2f}x, vs onehilb: {best_o/best_r:.2f}x)")

maxdiff_rfft = 0
for name, u in profs.items():
    a = onehilb(u, x); b = onehilb_rfft(u, x)
    maxdiff_rfft = max(maxdiff_rfft, np.max(np.abs(a-b)))
w(f"\nmax|onehilb - onehilb_rfft| across all profiles at N=2^14: {maxdiff_rfft:.3e}  (agreement <1e-12: {maxdiff_rfft<1e-12})")

json.dump({"t1": t1_results, "rel_rms": rel_rms_all,
           "t2_timing": {"dihiti":[best_d,med_d], "onehilb":[best_o,med_o], "onehilb_rfft":[best_r,med_r]},
           "label_mismatch_full": label_mismatch_full, "label_mismatch_interior": label_mismatch_interior,
           "maxdiff_rfft": maxdiff_rfft},
          open("t1_t2_results.json","w"), indent=2)

with open("t1_t2_output.txt","w") as f:
    f.write("\n".join(out))
print("DONE T1/T2")
