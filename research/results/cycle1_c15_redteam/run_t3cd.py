import numpy as np
import scipy, timeit, json
from bench import dihiti, onehilb, profiles_at_N, sign_min_interior, L, A, logistic_u, p2_u

out = []
def w(s=""):
    print(s)
    out.append(s)

# ---------------- T3(c) end-to-end timing at N=2^14 ----------------
w("## T3(c) end-to-end decimated-test timing at N=2^14\n")
N = 2**14
x, profs = profiles_at_N(N)
u_test = profs["P1_logistic_w0.05"]

# baseline from T1 at N=2^14 (re-measure here for apples-to-apples in same run)
def run_baseline():
    return dihiti(u_test, x).min()
t_base = timeit.repeat(run_baseline, repeat=20, number=1)
best_base = np.min(t_base)*1e6

w("| n (decimated) | best time (us) | speedup vs N=2^14 baseline |")
w("|---|---|---|")
decim_results = {}
for n in [16, 32, 64, 128, 256, 512, 1024, 2048]:
    step = N // n
    def run_decim(step=step):
        idx = np.arange(0, N, step)
        xu = x[idx]  # touches the input -- subsampling cost included
        uu = u_test[idx]
        v = onehilb(uu, xu)
        mask = np.abs(xu) < 0.99*L
        return v[mask].min()
    t = timeit.repeat(run_decim, repeat=20, number=1)
    best = np.min(t)*1e6
    speedup = best_base/best
    decim_results[n] = {"best_us": best, "speedup": speedup}
    w(f"| {n} | {best:.1f} | {speedup:.2f}x |")

w(f"\nBaseline (full dihiti, N=2^14): best {best_base:.1f} us")
w(f">=3x speedup achieved at n<=... : " + ", ".join(str(n) for n,r in decim_results.items() if r['speedup']>=3.0) or "NONE")

# ---------------- T3(d) honesty check ----------------
w("\n## T3(d) honesty check: near-threshold P2 alpha=0.8, 0.9 -- decimation sign errors\n")
N_GT = 2**16
x_gt = np.linspace(-L, L, N_GT, endpoint=True)
w("| alpha | n | sign | interior min | matches GT sign? |")
w("|---|---|---|---|---|")
for alpha in [0.8, 0.9]:
    u_gt = p2_u(x_gt, alpha)
    v_gt = onehilb(u_gt, x_gt)
    s_gt, m_gt = sign_min_interior(v_gt, x_gt)
    for n in [16,32,64,128,256,512,1024,2048,4096]:
        step = N_GT//n
        idx = np.arange(0, N_GT, step)
        xu = x_gt[idx]; uu = u_gt[idx]
        v = onehilb(uu, xu)
        s, m = sign_min_interior(v, xu)
        match = (s==s_gt)
        w(f"| {alpha} | {n} | {int(s)} | {m:.4e} | {match} |")

# adversarial: add high-frequency cosine near decimation Nyquist
w("\n### Adversarial profile: base P1 logistic w=0.05 + a*cos(m*pi*x/L)\n")
w("Testing whether decimation misses a high-frequency ripple that flips passivity locally.\n")
w("| m | a | n | sign(decim) | interior min (decim) | sign(GT @ N=2^16) | interior min (GT) | decimation correct? |")
w("|---|---|---|---|---|---|---|---|")
for n_target in [64, 128, 256]:
    # Nyquist of decimated grid corresponds to spatial frequency k_nyq = pi*n_target/(2L) roughly;
    # choose m such that cos(m pi x/L) has ~ n_target/2 periods over [-L,L] -> near-Nyquist for that n
    m = n_target // 2
    for a in [0.02, 0.05, 0.1]:
        def u_adv(x, m=m, a=a):
            return logistic_u(x, 0.05) + a*A*np.cos(m*np.pi*x/L)
        u_gt = u_adv(x_gt)
        v_gt = onehilb(u_gt, x_gt)
        s_gt, m_gt_val = sign_min_interior(v_gt, x_gt)

        step = N_GT // n_target
        idx = np.arange(0, N_GT, step)
        xu = x_gt[idx]; uu = u_adv(xu)
        v = onehilb(uu, xu)
        s, m_val = sign_min_interior(v, xu)
        correct = (s==s_gt)
        w(f"| {m} | {a} | {n_target} | {int(s)} | {m_val:.4e} | {int(s_gt)} | {m_gt_val:.4e} | {correct} |")

with open("t3cd_output.txt","w") as f:
    f.write("\n".join(out))
print("DONE T3cd")
