#!/usr/bin/env python3
"""
h9_domination_check.py -- numerical verification for research/theory/h9_potential_domination.md

Tests the claim that the DiHiTI passivity criterion is ONE potential-domination
relation, and that the corpus theorems are corollaries of it.

Sections:
  A. omega = Cauchy = Bal(delta_i, R) = pushforward of window Lebesgue; potential identity
  B. exact gap identity  pi*eps''(x) = -D*Delta(eta),  Delta(eta)=int ln|eta-xi| dP - (1/2)ln(1+eta^2)
  C. verdict agreement on the labelled corpus (profile_generator)
  D. two-atom exact law:  balanced pair at t+, -t-  passive  <=>  t+ + t- <= L
  E. asymmetric-interval generalisation of the middle-half theorem  (supp in [-a,b], a+b<=L)
  F. node collapse: n atoms => <= n candidate eta, roots of an explicit degree-n polynomial
  G. no fixed finite node set can work
  H. eta->infinity expansion reproduces the necessary cascade + perturbative depth
"""
import sys, os, time
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid

_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'theory'))
sys.path.insert(0, os.path.join(_ROOT, 'theory', 'passivity_2026Aug'))

L = 1.0
def dihiti(eps_re, x):
    u = np.gradient(eps_re, x)
    v = np.imag(hilbert(u))
    e = cumulative_trapezoid(v, x, initial=0.0)
    return e - np.linspace(e[0], e[-1], len(x))

def K(x, t, L=L):
    """atom kernel: pi*eps''(x) = int g(t) K(x;t) dt"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(np.cos(np.pi*t/(2*L))) - np.log(np.abs(np.sin(np.pi*(x-t)/(2*L))))

def xi(t, L=L):  return np.tan(np.pi*t/(2*L))

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return ok

RES = []
def rec(name, ok, detail=""):
    RES.append(check(name, ok, detail))

# ---------------------------------------------------------------- A
print("== A. omega = Cauchy: potential identity  int ln|eta-xi| dw(xi) = (1/2)ln(1+eta^2) ==")
# high-accuracy: substitute xi = tan(s), s in (-pi/2,pi/2), dw = ds/pi
Ns = 2_000_001
s = np.linspace(-np.pi/2, np.pi/2, Ns)[1:-1]
xs = np.tan(s)
err = []
for eta in [0.0, 0.3, 1.0, 2.5, 7.0, -4.0]:
    val = np.trapezoid(np.log(np.abs(eta-xs)), s)/np.pi
    err.append(abs(val - 0.5*np.log(1+eta**2)))
rec("Cauchy potential = (1/2)ln(1+eta^2)", max(err) < 2e-5, f"max abs err {max(err):.2e}")

# omega is the pushforward of normalised Lebesgue on the window under xi=tan(pi t/2L)
Nt = 2_000_001
tt = np.linspace(-L, L, Nt)[1:-1]
for eta in [0.0, 1.7]:
    val = np.trapezoid(np.log(np.abs(eta - xi(tt))), tt)/(2*L)
    rec(f"pushforward of window Lebesgue, eta={eta}", abs(val-0.5*np.log(1+eta**2)) < 2e-5,
        f"{val:.6f} vs {0.5*np.log(1+eta**2):.6f}")

# ---------------------------------------------------------------- B
print("== B. exact gap identity  pi*eps''(x) = -D*Delta(eta(x)) ==")
def gap_from_g(g, t, eta, L=L):
    """Delta(eta) = int ln|eta-xi(t)| g dt  -  (D/2) ln(1+eta^2)   (unnormalised: g, not P)"""
    D = np.trapezoid(g, t)
    xg = xi(t, L)
    out = np.empty_like(np.atleast_1d(eta), float)
    for i, e in enumerate(np.atleast_1d(eta)):
        out[i] = np.trapezoid(np.log(np.abs(e - xg))*g, t)
    return out - 0.5*D*np.log(1+np.atleast_1d(eta)**2)

nb = 1.7
def logistic(x, w): return 1.0 + (nb**2-1)/(1.0+np.exp(x/w))
Nx = 4001
x = np.linspace(-L, L, Nx)
ep = logistic(x, 0.08)
g  = -np.gradient(ep, x)
# interior points only (eta = +-inf at the seam)
m = np.abs(x) < 0.97*L
eta = xi(x[m])
lhs = -gap_from_g(g, x, eta)                       # = pi*eps''
ei  = dihiti(ep, x)
rel = np.max(np.abs(lhs - np.pi*ei[m]))/np.max(np.pi*ei)
rec("gap identity vs DiHiTI (logistic)", rel < 5e-3, f"max rel err {rel:.2e}")

# ---------------------------------------------------------------- C
print("== C. verdict agreement on the labelled corpus ==")
try:
    import profile_generator as pg
    rng = np.random.default_rng(7)
    agree = 0; tot = 0; disagree = []
    for smp in pg.sample(240, rng=rng):
        xs_ = smp['x']; Ls = xs_[-1]
        gs = -np.gradient(smp['e_re'], xs_)
        mm = np.abs(xs_) < 0.97*Ls
        et = np.tan(np.pi*xs_[mm]/(2*Ls))
        xg = np.tan(np.pi*xs_/(2*Ls))
        Ds = np.trapezoid(gs, xs_)
        dl = np.array([np.trapezoid(np.log(np.abs(e-xg))*gs, xs_) for e in et]) \
             - 0.5*Ds*np.log(1+et**2)
        scale = max(smp['e_im'].max(), 1e-12)
        v_dom = (dl.max() <= 1e-3*scale*np.pi)
        v_dih = smp['passive']
        tot += 1
        if v_dom == v_dih: agree += 1
        else: disagree.append((smp['name'], smp['min_eim'], -dl.max()/np.pi))
    rec("corpus verdict agreement", agree == tot, f"{agree}/{tot}; disagreements {disagree[:5]}")
except Exception as e:
    print(f"  [SKIP] corpus test: {e}")

# ---------------------------------------------------------------- D
print("== D. two-atom exact law:  passive <=> t+ + t- <= L ==")
def pair_gap_max(tp, tm, L=L):
    """balanced 2-atom: atoms at tp>0 and -tm<0, weights forced by M1=0.
    closed form: max_eta Delta = tanA ln(sinA/cosB) + tanB ln(sinB/cosA), A=pi tp/2L, B=pi tm/2L
    (normalised to total mass beta+gamma)."""
    A, B = np.pi*tp/(2*L), np.pi*tm/(2*L)
    return np.tan(A)*np.log(np.sin(A)/np.cos(B)) + np.tan(B)*np.log(np.sin(B)/np.cos(A))

def pair_min_eps(tp, tm, L=L, Nx=200001):
    """direct kernel evaluation of min eps'' for the balanced pair (mass-normalised)."""
    b, gm = np.tan(np.pi*tp/(2*L)), np.tan(np.pi*tm/(2*L))
    w1, w2 = gm/(b+gm), b/(b+gm)          # weights giving M1=0
    xx = np.linspace(-L, L, Nx)
    v = (w1*K(xx, tp, L) + w2*K(xx, -tm, L))/np.pi
    return np.nanmin(v)

ok = True; rows = []
for tp, tm in [(0.5,0.5),(0.6,0.4),(0.75,0.25),(0.9,0.1),(0.333,0.667),
               (0.52,0.52),(0.7,0.4),(0.8,0.3),(0.45,0.45),(0.2,0.2)]:
    pred_pass = (tp+tm) <= L + 1e-12
    mn = pair_min_eps(tp, tm)
    got_pass = mn > -1e-6
    # closed form max-gap, normalised the same way (divide by beta+gamma, then by pi)
    b, gm = np.tan(np.pi*tp/(2*L)), np.tan(np.pi*tm/(2*L))
    cf = -pair_gap_max(tp, tm)/((b+gm)*np.pi)
    ok &= (pred_pass == got_pass) and abs(cf - min(mn,0.0)) < 5e-4 if not pred_pass else (pred_pass == got_pass)
    rows.append((tp, tm, tp+tm, mn, cf, pred_pass, got_pass))
for r in rows:
    print(f"    t+={r[0]:.3f} t-={r[1]:.3f} sum={r[2]:.3f}  min_eps''={r[3]:+.5f} "
          f"closedform={r[4]:+.5f}  pred={'P' if r[5] else 'G'} got={'P' if r[6] else 'G'}")
rec("two-atom law t+ + t- <= L", ok)

# bisection on the threshold at fixed ratio
for ratio in [1.0, 2.0, 5.0]:
    lo, hi = 0.1, 1.9
    for _ in range(50):
        S = 0.5*(lo+hi)
        tp = S*ratio/(1+ratio); tm = S/(1+ratio)
        if tp >= 0.999*L or tm >= 0.999*L: hi = S; continue
        if pair_min_eps(tp, tm, Nx=60001) < -1e-9: hi = S
        else: lo = S
    rec(f"threshold sum at ratio {ratio}", abs(0.5*(lo+hi) - L) < 5e-3, f"S* = {0.5*(lo+hi):.5f} vs L=1")

# ---------------------------------------------------------------- E
print("== E. asymmetric-interval generalisation:  supp g in [-a,b], a+b <= L, M1=0 ==")
Ne = 2**13
xe = np.linspace(-L, L, Ne)
tanx = xi(xe)
def eps_of(gv, xg=None):
    xg = xe if xg is None else xg
    G = cumulative_trapezoid(gv, xg, initial=0.0)
    return 1.0 + G[-1] - G
def balance(gv, mask):
    for _ in range(300):
        M1 = np.trapezoid(gv*tanx, xe)
        nrm = np.trapezoid(np.abs(gv), xe)
        if nrm == 0 or abs(M1) < 1e-13*nrm: break
        M2 = np.trapezoid(gv*tanx**2, xe)
        if M2 <= 0: break
        gv = np.clip(gv*(1 - (M1/M2)*tanx), 0, None)
        gv[~mask] = 0.0
    return gv

def mc_interval(a, b, ntrial=200, seed=0):
    mask = (xe >= -a) & (xe <= b)
    worst = np.inf; wparams = None
    for i in range(ntrial):
        r = np.random.default_rng(seed+i)
        gv = np.zeros(Ne)
        for _ in range(r.integers(2, 7)):
            c = r.uniform(-a, b); sw = r.uniform(0.01, 0.20); amp = r.uniform(0.1, 1)
            gv += amp*np.exp(-0.5*((xe-c)/sw)**2)
        gv[~mask] = 0.0
        gv = balance(gv, mask)
        Dv = np.trapezoid(gv, xe)
        if Dv < 1e-6: continue
        mn = dihiti(eps_of(gv), xe).min()/Dv
        if mn < worst: worst = mn; wparams = i
    return worst

for (a, b) in [(0.5,0.5), (0.25,0.75), (0.1,0.9), (0.4,0.6), (0.75,0.25)]:
    w = mc_interval(a, b, 200, seed=100)
    rec(f"MC balanced on [-{a},{b}] (a+b={a+b:.2f}<=1)", w >= -2e-8, f"worst min/D = {w:.2e}")
# and beyond: should be able to fail
fails = []
for (a, b) in [(0.55,0.55), (0.3,0.85), (0.2,0.95)]:
    w = mc_interval(a, b, 300, seed=200)
    fails.append((a, b, w))
    print(f"    [info] a+b={a+b:.2f} -> worst min/D = {w:.3e}")
rec("a+b>L admits failures (necessity of the interval bound)",
    any(f[2] < -1e-4 for f in fails), f"{fails}")

# ---------------------------------------------------------------- F
print("== F. node collapse for atomic drop measures ==")
def atomic_nodes(ts, ws, L=L):
    """critical points of Delta: roots of Q(eta) = sum_j w_j prod_{k!=j}(eta-xi_k)(1+eta^2)
                                                   - D*eta*prod_k(eta-xi_k).   deg Q <= n."""
    xs_ = xi(np.asarray(ts, float), L); ws = np.asarray(ws, float); D = ws.sum()
    n = len(xs_)
    prod = np.poly(xs_)                       # coeffs of prod (eta - xi_k), degree n
    Q = np.zeros(n+2)
    for j in range(n):
        pj = np.poly(np.delete(xs_, j))       # degree n-1
        term = np.polymul(pj, [1.0, 0.0, 1.0])   # * (1+eta^2), degree n+1
        Q[-len(term):] += ws[j]*term
    term2 = np.polymul(prod, [D, 0.0])        # D*eta*prod, degree n+1
    Q[-len(term2):] -= term2
    Q = np.trim_zeros(Q, 'f')
    if len(Q) <= 1: return np.array([])
    r = np.roots(Q)
    return np.real(r[np.abs(r.imag) < 1e-8*(1+np.abs(r.real))])

def atomic_gap(ts, ws, eta, L=L):
    xs_ = xi(np.asarray(ts, float), L); D = np.sum(ws)
    eta = np.atleast_1d(np.asarray(eta, float))
    with np.errstate(divide='ignore'):
        val = np.array([np.sum(ws*np.log(np.abs(e - xs_))) for e in eta])
    return val - 0.5*D*np.log(1+eta**2)

def atomic_min_eps_exact(ts, ws, L=L):
    nd = atomic_nodes(ts, ws, L)
    if len(nd) == 0: return 0.0
    return -np.max(np.append(atomic_gap(ts, ws, nd, L), 0.0))/np.pi

def atomic_min_eps_grid(ts, ws, L=L, Nx=400001):
    xx = np.linspace(-L, L, Nx)
    v = np.zeros(Nx)
    for t0, w0 in zip(ts, ws): v += w0*K(xx, t0, L)
    return np.nanmin(v)/np.pi

rng = np.random.default_rng(3)
errs = []; ndeg = []
for trial in range(300):
    n = int(rng.integers(1, 9))
    ts = np.sort(rng.uniform(-0.95*L, 0.95*L, n))
    ws = rng.uniform(0.05, 1.0, n)
    if rng.random() < 0.5:   # balance M1 = 0 (interesting near-threshold cases)
        xs_ = xi(ts)
        if np.any(xs_ > 0) and np.any(xs_ < 0):
            ws = ws*np.where(xs_ > 0, np.abs(xs_[xs_ < 0]).mean(), np.abs(xs_[xs_ > 0]).mean())
            # simple exact rebalance: scale positive-side weights
            sp = np.sum(ws[xs_ > 0]*xs_[xs_ > 0]); sn = -np.sum(ws[xs_ < 0]*xs_[xs_ < 0])
            if sp > 0 and sn > 0: ws = np.where(xs_ > 0, ws*sn/sp, ws)
    a = atomic_min_eps_exact(ts, ws); b_ = atomic_min_eps_grid(ts, ws)
    errs.append(abs(a-b_)); ndeg.append(len(atomic_nodes(ts, ws)) - n)
rec("node-collapse exact vs 4e5-grid direct sum", max(errs) < 2e-4,
    f"max |diff| = {max(errs):.2e} over 300 random atomic profiles")
rec("node count <= n", max(ndeg) <= 0, f"max (#nodes - n) = {max(ndeg)}")

# timing: n atoms, grid needed to resolve the min vs node test
n = 40
ts = np.sort(np.random.default_rng(11).uniform(-0.9*L, 0.9*L, n)); ws = np.ones(n)/n
t0 = time.perf_counter()
for _ in range(20): atomic_min_eps_exact(ts, ws)
t_node = (time.perf_counter()-t0)/20
Ngrid = 2**14
xx = np.linspace(-L, L, Ngrid)
ep_step = np.zeros(Ngrid)
for tj, wj in zip(ts, ws): ep_step += wj*(xx < tj)
t0 = time.perf_counter()
for _ in range(20): dihiti(ep_step, xx)
t_fft = (time.perf_counter()-t0)/20
print(f"    n={n} atoms: node test {t_node*1e3:.3f} ms   |   DiHiTI at N=2^14 {t_fft*1e3:.3f} ms"
      f"   speedup {t_fft/t_node:.1f}x")

# ---------------------------------------------------------------- G
print("== G. no fixed finite node set suffices ==")
Ng = 2**13
xg = np.linspace(-L, L, Ng, endpoint=False)
nodes_x = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])*L        # any 5 prescribed nodes
xstar = 0.62*L
v = np.exp(-0.5*((xg-0.1*L)/0.30)**2) - 0.35*np.exp(-0.5*((xg-xstar)/0.02)**2)
v = v*(1+np.cos(np.pi*xg/L))/2                             # force v(+-L)=0
Dv = 2.0
ep_adv = 3.0 - Dv*xg/(2*L) - np.imag(hilbert(v))
v_back = dihiti(ep_adv, xg)
at_nodes = np.interp(nodes_x, xg, v_back)
rec("adversary: passes all 5 prescribed nodes, fails between",
    np.all(at_nodes > 0) and v_back.min() < -1e-3,
    f"node values {np.round(at_nodes,4)}, global min {v_back.min():.4f}")

# ---------------------------------------------------------------- H
print("== H. eta->infinity expansion reproduces the cascade + depth law ==")
for x0 in [0.02, 0.05, 0.10]:
    epo = logistic(x - x0*L, 0.05)
    go = -np.gradient(epo, x)
    mm = np.abs(x) < 0.98*L
    M1 = np.trapezoid((go*np.tan(np.pi*x/(2*L)))[mm], x[mm])
    M2 = np.trapezoid((go/np.cos(np.pi*x/(2*L))**2)[mm], x[mm])
    pred = -M1**2/(2*np.pi*M2)
    got = dihiti(epo, x).min()
    # the same numbers as the max of the two-term expansion of Delta
    eta_star = -M2/M1
    got_exp = -np.max(gap_from_g(go, x, np.array([eta_star])))/np.pi
    rec(f"depth law from Delta-expansion, x0={x0}L",
        abs(pred-got) < 0.06*abs(got)+1e-5, f"pred {pred:.5f} DiHiTI {got:.5f} Delta@eta* {got_exp:.5f}")
# single atom -> Cor 4.4 in one line
for x0 in [0.2, 0.5]:
    b = xi(x0*L); pred = -np.log(np.sqrt(1+b**2))/np.pi
    got = np.nanmin(K(np.linspace(-L, L, 400001), x0*L))/np.pi
    rec(f"single atom (Cor 4.4) x0={x0}L", abs(pred-got) < 1e-5, f"{pred:.6f} vs {got:.6f}")

print(f"\n==== {sum(RES)}/{len(RES)} checks passed ====")
