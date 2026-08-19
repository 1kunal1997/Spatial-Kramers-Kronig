import numpy as np
import scipy, timeit, json
from bench import (dihiti, onehilb, profiles_at_N, sign_min_interior, L, A)

out = []
def w(s=""):
    print(s)
    out.append(s)

N_GT = 2**16
x_gt, profs_gt = profiles_at_N(N_GT)

# ground truth verdicts (sign + value of interior min), using onehilb on full N
gt = {}
for name, u in profs_gt.items():
    v = onehilb(u, x_gt)
    s, m = sign_min_interior(v, x_gt)
    gt[name] = (s, m)

w("## T3(a) decimation: smallest n matching sign / matching value to 10%\n")
w("| profile | N | n_sign | n_10pct | GT interior min |")
w("|---|---|---|---|---|")

ns_try = [2**k for k in range(4, 17)]  # 16 ... 65536
t3a = {}
for name in profs_gt:
    s_gt, m_gt = gt[name]
    n_sign = None
    n_10pct = None
    for n in ns_try:
        step = N_GT // n
        if step < 1 or N_GT % n != 0:
            continue
        idx = np.arange(0, N_GT, step)
        xu = x_gt[idx]
        uu = profs_gt[name][idx]
        v = onehilb(uu, xu)
        s, m = sign_min_interior(v, xu)
        if n_sign is None and s == s_gt:
            n_sign = n
        if n_10pct is None and m_gt != 0 and abs(m - m_gt) <= 0.1*abs(m_gt):
            n_10pct = n
        if n_sign is not None and n_10pct is not None:
            break
    t3a[name] = {"n_sign": n_sign, "n_10pct": n_10pct, "gt_min": m_gt, "gt_sign": int(s_gt)}
    w(f"| {name} | {N_GT} | {n_sign} | {n_10pct} | {m_gt:.4e} |")

json.dump(t3a, open("t3a_results.json","w"), indent=2)

# ---------------- T3(b) spectral decay ----------------
w("\n## T3(b) Fourier spectrum decay of periodized p = u + (D/2L)x\n")
w("| profile | n(tail<1e-6) | n(tail<1e-10) |")
w("|---|---|---|")
t3b = {}
for name, u in profs_gt.items():
    x = x_gt
    D = u[0]-u[-1]
    p = u + (D/(2*L))*x
    P = np.fft.rfft(p)
    power = np.abs(P)**2
    total = np.sum(power)
    cumtail = total - np.cumsum(power)  # tail energy beyond mode k
    frac_tail = cumtail/total
    n6 = None; n10 = None
    for k in range(1, len(frac_tail)):
        if n6 is None and frac_tail[k] < 1e-6:
            n6 = k
        if n10 is None and frac_tail[k] < 1e-10:
            n10 = k
        if n6 is not None and n10 is not None:
            break
    t3b[name] = {"n_tail_1e-6": n6, "n_tail_1e-10": n10}
    w(f"| {name} | {n6} | {n10} |")

json.dump(t3b, open("t3b_results.json","w"), indent=2)

with open("t3ab_output.txt","w") as f:
    f.write("\n".join(out))
print("DONE T3ab")
