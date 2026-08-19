"""E1(a) v3 -- the SOUND certificate the cycle-1 report asked for (E3), built correctly.

K13 died because it used repeated CENTRED differences, whose symbol i*sin(theta)/h VANISHES at
theta=pi (m=N/2): it under-estimates exactly the Nyquist modes it must bound.  FORWARD differences
do not have that defect: the symbol of (p_{j+1}-p_j)/h is (e^{i th}-1)/h, modulus 2|sin(th/2)|/h,
and the ratio to the true |m|pi/L is 2 sin(th/2)/th = sinc(th/2) in [2/pi, 1] for |th|<=pi.
Hence, SOUNDLY, for the k-th repeated forward difference D_h^k,
        ||p^(k)||_2  <=  (pi/2)^k * ||D_h^k p||_2 / h^k                        (B1)
with NO under-estimation.  Then Cauchy-Schwarz on the tail (M = n/2):
        S_M := sum_{|m|>=M} |phat_m|
             <= (L/pi)^k ||p^(k)||_2 * sqrt( 2 / ((2k-1) M^(2k-1)) )           (B2)
and the decimation error in eps'' obeys
        || eps''_dec - eps'' ||_inf  <=  4 S_M                                 (B3)
   (aliasing folds the tail into |j|<M: |qhat_j - phat_j| <= tail, plus the discarded tail
    itself, giving 2 S_M; the 'subtract the value at -L' gauge doubles it again.)

CERTIFICATE:  if |min_dec| > 4 S_M  then the SIGN of the verdict is certified.
This is TWO-SIDED (certifies PASS and FAIL) and costs O(N) adds + O(n log n).

This script (i) checks B1/B2/B3 numerically against the true spectral quantities -- soundness is
VERIFIED, not assumed -- and (ii) reports the safe-and-fast table.
"""
import numpy as np, timeit, json, os
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid

OUT = os.path.dirname(os.path.abspath(__file__))
nb = 1.7; L = 1.0; A = nb**2 - 1; FRAC = 0.95

def dihiti(u, x):
    du = np.gradient(u, x); v = np.imag(hilbert(du))
    e = cumulative_trapezoid(v, x, initial=0.0)
    return e - np.linspace(e[0], e[-1], len(x))

def imin(v, x, frac=FRAC):
    Lx = 0.5*(x[-1]-x[0]); return v[np.abs(x) <= frac*Lx].min()

def ramp(u, x):
    L_ = 0.5*(x[-1]-x[0]); D = u[0]-u[-1]; return u + (D/(2*L_))*x

def tail_bound(u, x, M, k=4):
    """SOUND upper bound on sum_{|m|>=M}|phat_m|, O(N), forward differences only."""
    N = len(x); L_ = 0.5*(x[-1]-x[0]); h = 2*L_/(N-1)
    p = ramp(u, x)[:-1]                      # one period, N-1 samples
    d = p.copy()
    for _ in range(k):
        d = np.roll(d, -1) - d               # periodic forward difference
    Dk = np.sqrt(np.mean(d**2))/h**k         # ||D_h^k p||_2  (normalised L2 on the period)
    nrm = (np.pi/2)**k * Dk                  # >= ||p^(k)||_2   (B1)
    return (L_/np.pi)**k * nrm * np.sqrt(2.0/((2*k-1)*M**(2*k-1)))

def true_tail(u, x, M):
    """exact sum_{|m|>=M}|phat_m| from the full-N FFT (reference only; O(N log N))."""
    p = ramp(u, x)[:-1]; c = np.fft.fft(p)/len(p)
    m = np.fft.fftfreq(len(p), d=1.0/len(p))
    return np.abs(c[np.abs(m) >= M]).sum()

def build_cases(x, nrand=60, seed=11):
    c = {}; kind = {}
    for w in [0.3, 0.15, 0.05, 0.02, 0.005]:
        c[f"logistic w={w}"] = 1 + A/(1+np.exp(x/w)); kind[f"logistic w={w}"] = 'phys'
    for al in [0.0,0.3,0.6,0.6640,0.6662,0.6680,0.8,0.9,1.0,1.5]:
        k = f"bumps a={al}"
        c[k] = (1 + A/(1+np.exp(x/0.05))
                + al*A*(np.exp(-((x+0.45)/0.22)**2) - np.exp(-((x-0.45)/0.22)**2)))
        kind[k] = 'phys'
    rng = np.random.default_rng(seed)
    for i in range(nrand):
        g = sum(rng.uniform(.1,1)*np.exp(-((x-rng.uniform(-.9,.9))/rng.uniform(.02,.3))**2)
                for _ in range(rng.integers(3,7)))
        G = cumulative_trapezoid(g, x, initial=0); u = 1+G[-1]-G
        c[f"rand{i}"] = 1 + (u-1)*A/(u[0]-u[-1]); kind[f"rand{i}"] = 'phys'
    for m_ in [63,64,65,127,128,129,255,256,257,511,512,513,1024,2048,4096]:
        k = f"cos m={m_}"
        c[k] = 1 + A/(1+np.exp(x/0.05)) + 0.02*A*np.cos(m_*np.pi*x/L); kind[k] = 'adv'
    return c, kind

def main(N=2**16, ns=(256,512,1024,2048,4096), ks=(2,3,4,6)):
    x = np.linspace(-L, L, N); cases, kind = build_cases(x); keys = list(cases)

    # ---------- (i) SOUNDNESS AUDIT of B1/B2/B3 ----------
    print("=== soundness audit: bound / truth  (must be >= 1 for every row) ===")
    print(f"{'case':16s} {'M':>6s} {'k':>2s} {'true_tail':>11s} {'bound':>11s} {'ratio':>9s} "
          f"{'true_err':>10s} {'4*bound':>10s} {'ok':>4s}")
    audit = []; bad = 0
    for k in ['logistic w=0.05','logistic w=0.3','bumps a=1.0','rand3','cos m=257','cos m=1024']:
        u = cases[k]; D = abs(u[0]-u[-1]); tr = imin(dihiti(u, x), x)
        for n in (512, 2048):
            M = n//2; s = N//n
            dv = imin(dihiti(u[::s], x[::s]), x[::s])
            err = abs(dv-tr)/D
            tt = true_tail(u, x, M)/D
            for kk in ks:
                b = tail_bound(u, x, M, kk)/D
                ok = (b >= tt*0.999) and (4*b >= err*0.999)
                bad += not ok
                print(f"{k:16s} {M:6d} {kk:2d} {tt:11.3e} {b:11.3e} {b/max(tt,1e-300):9.2e} "
                      f"{err:10.3e} {4*b:10.3e} {'ok' if ok else 'BAD':>4s}")
                audit.append(dict(case=k, M=M, k=kk, true_tail=tt, bound=b, err=err, ok=bool(ok)))
    print(f"soundness violations: {bad}")

    # ---------- (ii) certificate coverage + wall clock ----------
    print("\n=== certificate: |min_dec| > 4*bound  =>  sign certified ===")
    u0 = cases['logistic w=0.05']
    T = lambda f: min(timeit.repeat(f, number=10, repeat=7))/10
    t_B0 = T(lambda: imin(dihiti(u0, x), x))
    from e1a import onehilb
    t_B1 = T(lambda: imin(onehilb(u0, x), x))
    print(f"B0 {t_B0*1e6:.1f}us  B1 {t_B1*1e6:.1f}us")
    rows = []
    print(f"{'n':>6s} {'k':>2s} {'cert%':>7s} {'phys cert%':>11s} {'adv cert%':>10s} "
          f"{'false':>6s} {'t_bnd':>8s} {'t_dec':>8s} {'E[t]':>9s} {'vsB0':>6s} {'vsB1':>6s}")
    for n in ns:
        M = n//2; s = N//n
        tdec = T(lambda s=s: imin(dihiti(u0[::s], x[::s]), x[::s]))
        for kk in ks:
            tbnd = T(lambda kk=kk, M=M: tail_bound(u0, x, M, kk))
            nc = ncp = nca = 0; nf = 0; npz = nadv = 0
            for key in keys:
                u = cases[key]; D = abs(u[0]-u[-1])
                if D < 1e-12: continue
                tr = imin(dihiti(u, x), x); dv = imin(dihiti(u[::s], x[::s]), x[::s])
                b = 4*tail_bound(u, x, M, kk)
                cert = abs(dv) > b
                if kind[key] == 'phys': npz += 1; ncp += cert
                else: nadv += 1; nca += cert
                nc += cert
                if cert and ((tr >= 0) != (dv >= 0)): nf += 1
            tot = npz+nadv; pc = nc/tot
            Et = tbnd + tdec + (1-pc)*t_B1
            print(f"{n:6d} {kk:2d} {100*pc:6.1f}% {100*ncp/npz:10.1f}% {100*nca/max(nadv,1):9.1f}% "
                  f"{nf:6d} {tbnd*1e6:7.1f}u {tdec*1e6:7.1f}u {Et*1e6:8.1f}u "
                  f"{t_B0/Et:6.2f} {t_B1/Et:6.2f}")
            rows.append(dict(n=n, k=kk, cert=pc, cert_phys=ncp/npz, cert_adv=nca/max(nadv,1),
                             false=nf, t_bnd=tbnd, t_dec=tdec, Et=Et, vsB0=t_B0/Et, vsB1=t_B1/Et))
    json.dump(dict(N=N, audit=audit, rows=rows, t_B0=t_B0, t_B1=t_B1),
              open(os.path.join(OUT, f"e1a3_N{N}.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
