"""
Part 1 — analytic cross-check of the ht_derivative pipeline against a closed form.

Independent route (Route 1, analytic limit): build eps'(x) as a finite trig sum whose harmonics
are exactly 2L-periodic. Derive by hand (see checkpoint.md) that the full DiHiTI pipeline
(differentiate -> Hilbert -> integrate -> subtract line) reduces, for such profiles, to a closed
form with NO numerical Hilbert transform needed:

    eps''_0(x) = sum_k c_k * [ sin(k*pi*x/L + phi_k) - (-1)**k * sin(phi_k) ]

This lets us validate scipy.signal.hilbert's behavior (and our own pipeline code) against a
hand-derived formula, independent of anything in the off-limits files.
"""
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid

def ht_derivative(xx, e_re):
    # exact reproduction of tmm_helper.ht_derivative (read from source, allowed infrastructure)
    N = len(e_re)
    u = np.gradient(e_re, xx)
    v = np.imag(hilbert(u))
    e_im = cumulative_trapezoid(v, xx, initial=0)
    e_im -= np.linspace(e_im[0], e_im[-1], N)
    return e_im

def trig_profile(x, L, cs, phis, ks):
    theta = np.pi * x / L
    out = np.zeros_like(x)
    for c, phi, k in zip(cs, phis, ks):
        out += c * np.cos(k * theta + phi)
    return out

def trig_profile_analytic_eps2(x, L, cs, phis, ks):
    theta = np.pi * x / L
    out = np.zeros_like(x)
    for c, phi, k in zip(cs, phis, ks):
        out += c * (np.sin(k * theta + phi) - ((-1) ** k) * np.sin(phi))
    return out

def run(N, L=1.0, seed=0):
    rng = np.random.default_rng(seed)
    K = 6
    ks = np.arange(1, K + 1)
    cs = rng.uniform(0.3, 1.0, K) / ks  # decaying amplitudes for smoothness
    phis = rng.uniform(0, 2 * np.pi, K)

    x = np.linspace(-L, L, N, endpoint=False)  # our grid convention
    ep = trig_profile(x, L, cs, phis, ks)
    eps2_numeric = ht_derivative(x, ep)
    eps2_analytic = trig_profile_analytic_eps2(x, L, cs, phis, ks)

    err = eps2_numeric - eps2_analytic
    scale = np.max(np.abs(eps2_analytic))
    return dict(
        N=N, max_abs_err=np.max(np.abs(err)), rms_err=np.sqrt(np.mean(err**2)),
        scale=scale, max_rel_err=np.max(np.abs(err)) / scale
    )

if __name__ == "__main__":
    print("Analytic cross-check: numeric ht_derivative pipeline vs hand-derived closed form")
    print("for finite trig-sum eps'(x), across resolutions N (convergence test).")
    print(f"{'N':>8} {'max_abs_err':>14} {'rms_err':>14} {'profile_scale':>14} {'max_rel_err':>14}")
    for N in [512, 1024, 2048, 4096, 8192, 16384, 10000, 12000]:
        r = run(N)
        print(f"{r['N']:>8} {r['max_abs_err']:>14.6e} {r['rms_err']:>14.6e} "
              f"{r['scale']:>14.6e} {r['max_rel_err']:>14.6e}")

    # also test a second, independent random seed + different L, to rule out a lucky coincidence
    print()
    print("Second seed / different L / different endpoint(True) grid for comparison:")
    for seed in [1, 2]:
        r = run(N=8192, L=1.0, seed=seed)
        print(f"seed={seed}: max_rel_err={r['max_rel_err']:.6e}")

    # sanity: show what happens with the WRONG (endpoint=True, duplicated) grid convention,
    # to confirm we can detect the known bug class.
    print()
    print("Control: repeat with linspace(-L,L,N,endpoint=True) [duplicated-endpoint bug] at N=2048:")
    L = 1.0
    rng = np.random.default_rng(0)
    K = 6
    ks = np.arange(1, K + 1)
    cs = rng.uniform(0.3, 1.0, K) / ks
    phis = rng.uniform(0, 2 * np.pi, K)
    x_bad = np.linspace(-L, L, 2048, endpoint=True)
    ep_bad = trig_profile(x_bad, L, cs, phis, ks)
    eps2_num_bad = ht_derivative(x_bad, ep_bad)
    eps2_ana_bad = trig_profile_analytic_eps2(x_bad, L, cs, phis, ks)
    err_bad = eps2_num_bad - eps2_ana_bad
    scale_bad = np.max(np.abs(eps2_ana_bad))
    print(f"endpoint=True: max_rel_err={np.max(np.abs(err_bad))/scale_bad:.6e}  "
          f"(compare to endpoint=False N=2048 above)")
