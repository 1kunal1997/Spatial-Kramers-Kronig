"""Fixed-feature-width companions to patch4/patch11, for the reordered deck.

A)  Thickness sweep at FIVE angles (patch4 only ever solved theta=80).
    w = lam/5 fixed, tapered ell = 0.5 lam fixed, N = min(8000, max(400, 50*L/lam)),
    i.e. element SIZE held at 0.02 lam.                       -> P12_TH_{kind}_L{L}

B)  Element sweep at L = 100 lam with the SAME fixed width w = lam/5, so the
    equal-cost slide is tied to the truncation sweep above it rather than to the
    self-similar (w = L/80) family in patch11.                -> P12_EL_{kind}_N{n}

Geometry identical to patch9/patch11 so everything concatenates:
    sKK : L_low = 2 lam, N_low = 400.   PML reference comes from P9_PML_* / P10_PML_*.
"""
import os, sys, time, numpy as np, jpype, mph
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); import tmm_helper as tmm_h
MODEL = os.path.join(ROOT, 'Comsol Files', 'skk_master.mph')
OUT   = os.path.join(HERE, 'master_data.npz')

lam = 0.8e-6; EB = 1.7**2; WFIX = lam / 5; ELL = 0.5 * lam
ANG = '0 45 80 85 89'; THS = np.array([0., 45., 80., 85., 89.])
LG  = sorted(set(round(float(v), 3) for v in np.geomspace(1, 100, 21)))
r   = (2000 / 5) ** (1 / 23)
NE  = sorted(n for n in (set(int(round(v)) for v in np.geomspace(5, 2000, 24))
             | set([int(round(2000 * r**k)) for k in range(1, 6)]) | {8000}) if n >= 100)
NE += [int(round(8000 * r**k)) for k in (1, 2)] + [16000]   # w=lam/5 needs a finer mesh
NE = sorted(set(NE))


def profile(kind, L):
    """FIXED absolute feature width -- the patch4 recipe."""
    y = np.linspace(0.0, L, 6001)
    if kind == 'logistic':
        er = 1.0 + (EB - 1.0) / (1.0 + np.exp(-(y - L / 2) / WFIX))
        return y, er, -tmm_h.ht_derivative(y, er), 1.0
    u = y - L / 2; D = u**2 + WFIX**2; A = 4.0
    if kind == 'lorentzian':
        return y, EB + A * WFIX * u / D, A * WFIX**2 / D, EB
    if kind == 'tapered':
        return y, EB + A * WFIX * u / D / np.cosh(u / ELL), A * WFIX**2 / D, EB
    raise ValueError(kind)


t0 = time.time(); client = mph.start(); model = client.load(MODEL); jm = model.java
comp = jm.component('comp1'); pml = comp.coordSystem('pml1'); STUDY = model.studies()[0]
old = dict(np.load(OUT)); RES = {}; nd = [0]
jm.study('std1').feature('stat').set('plistarr', jpype.JArray(jpype.JString)([ANG]))
pml.active(False)
model.parameter('L_low', '2*lam'); model.parameter('N_low', '400')


def push(y, er, ei):
    comp.func('i_re').set('table', jpype.JArray(jpype.JString, 2)(
        [[repr(float(a)), repr(float(b))] for a, b in zip(y, er)]))
    comp.func('i_im').set('table', jpype.JArray(jpype.JString, 2)(
        [[repr(float(a)), repr(float(b))] for a, b in zip(y, ei)]))


def run(key):
    try:
        model.solve(STUDY)
        R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
    except Exception as e:
        print(f'  !! {key}: {str(e).splitlines()[0][:60]}', flush=True); return
    RES[key] = np.vstack([THS[:len(R)], R]); nd[0] += 1


print(f'=== A: thickness sweep, w=lam/5 fixed, 5 angles ({len(LG)*3} solves) ===', flush=True)
for L in LG:
    N = int(min(8000, max(400, 50 * L)))
    model.parameter('L_term', f'{L}*lam'); model.parameter('N_term', str(N))
    row = []
    for k in ['logistic', 'lorentzian', 'tapered']:
        y, er, ei, el = profile(k, L * lam); push(y, er, ei)
        model.parameter('eps_low', repr(float(el)))
        run(f'P12_TH_{k}_L{L}')
        v = RES.get(f'P12_TH_{k}_L{L}')
        row.append(v[1][2] if v is not None else np.nan)
    print(f'  L={L:7.2f} N={N:5d}  R(80): ' + '  '.join(f'{v:.3e}' for v in row), flush=True)

print(f'\n=== B: element sweep at L=100 lam, w=lam/5 fixed ({len(NE)*3} solves) ===', flush=True)
model.parameter('L_term', '100*lam')
for k in ['logistic', 'lorentzian', 'tapered']:
    y, er, ei, el = profile(k, 100 * lam); push(y, er, ei)
    model.parameter('eps_low', repr(float(el)))
    for N in NE:
        model.parameter('N_term', str(N)); run(f'P12_EL_{k}_N{N}')
    v = RES.get(f'P12_EL_{k}_N{NE[-1]}')
    if v is not None:
        print(f'  {k:11s} N={NE[-1]}: ' + '  '.join(f'{t:.0f}deg={x:.3e}'
              for t, x in zip(THS, v[1])), flush=True)

old.update(RES); np.savez(OUT, **old); client.clear()
print(f'\nDONE {nd[0]} solves in {time.time()-t0:.0f}s', flush=True)
