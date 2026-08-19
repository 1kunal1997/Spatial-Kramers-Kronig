"""
Part 4 — N-independence (consequence (c)). Repeat the "offset" (dilation-covariant) slope
measurement at a FIXED x0 and FIXED delta (in absolute units, delta = 0.02*L) across several N,
including non-power-of-2 values, to test whether the geometric coefficient C_phi is stable under
grid refinement. If ht_derivative's numerics (np.gradient + FFT hilbert + cumulative_trapezoid) were
secretly N-dependent in a way that happened to cancel only at powers of 2 (the exact failure mode
flagged in PROTOCOL.md), that would show up here as N in {non-power-of-2} disagreeing with N in
{power-of-2} while the power-of-2 values agree with each other.
"""
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
import json

def ht_derivative(xx, e_re):
    N = len(e_re)
    u = np.gradient(e_re, xx)
    v = np.imag(hilbert(u))
    e_im = cumulative_trapezoid(v, xx, initial=0)
    e_im -= np.linspace(e_im[0], e_im[-1], N)
    return e_im

def raised_cosine_bump(x, center, delta, a):
    s = (x - center) / (3 * delta)
    out = np.zeros_like(x)
    mask = np.abs(s) < 1
    out[mask] = a * 0.5 * (1 + np.cos(np.pi * s[mask]))
    return out

def base_profile(x, L, cs, phis, ks):
    theta = np.pi * x / L
    out = np.zeros_like(x)
    for c, phi, k in zip(cs, phis, ks):
        out += c * np.cos(k * theta + phi)
    return out

def measure(N, L=1.0, x0_target=0.0, delta=0.02, a_probe=1e-3, seed=7):
    rng = np.random.default_rng(seed)
    K = 6
    ks = np.arange(1, K + 1)
    cs = rng.uniform(0.3, 1.0, K) / ks
    phis = rng.uniform(0, 2 * np.pi, K)

    x = np.linspace(-L, L, N, endpoint=False)
    i0 = np.argmin(np.abs(x - x0_target))
    x0 = x[i0]
    ep0 = base_profile(x, L, cs, phis, ks)
    base_val = ht_derivative(x, ep0)[i0]
    center = x0 + delta / 2.0

    vals = []
    for a in [a_probe, 2 * a_probe, 4 * a_probe]:
        bump = raised_cosine_bump(x, center, delta, a)
        eps2 = ht_derivative(x, ep0 + bump)
        vals.append((eps2[i0] - base_val) / a)
    slope = np.mean(vals)
    lin_spread = (max(vals) - min(vals)) / abs(slope)
    C_phi = -1.0 / (np.pi * slope)
    return dict(N=N, x0=float(x0), dx=float(x[1] - x[0]), slope=float(slope),
                C_phi=float(C_phi), lin_spread=float(lin_spread))

if __name__ == "__main__":
    Ns = [4096, 8192, 16384, 10000, 12000, 15000]
    print(f"{'N':>8} {'is_pow2':>8} {'x0':>8} {'dx':>10} {'slope':>12} {'C_phi':>10} {'lin_spread':>12}")
    results = []
    for N in Ns:
        is_pow2 = (N & (N - 1)) == 0
        r = measure(N)
        r["is_pow2"] = bool(is_pow2)
        results.append(r)
        print(f"{r['N']:>8} {str(is_pow2):>8} {r['x0']:>8.4f} {r['dx']:>10.3e} "
              f"{r['slope']:>12.6f} {r['C_phi']:>10.6f} {r['lin_spread']:>12.3e}")

    C_phis = np.array([r["C_phi"] for r in results])
    print(f"\nC_phi across N in {Ns}: mean={C_phis.mean():.6f}  std={C_phis.std():.6f}  "
          f"rel_std={C_phis.std()/C_phis.mean()*100:.4f}%")
    pow2 = np.array([r["C_phi"] for r in results if r["is_pow2"]])
    nonpow2 = np.array([r["C_phi"] for r in results if not r["is_pow2"]])
    print(f"power-of-2 subset: mean={pow2.mean():.6f} std={pow2.std():.6f}")
    print(f"non-power-of-2 subset: mean={nonpow2.mean():.6f} std={nonpow2.std():.6f}")
    print(f"cross-group relative difference: "
          f"{abs(pow2.mean()-nonpow2.mean())/pow2.mean()*100:.4f}%")

    with open("research/results/cycle3_v2a_replication/part4_N_independence_data.json", "w") as f:
        json.dump(results, f, indent=2)
