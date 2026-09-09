"""Grazing-incidence equal-cost run: high PMLfactors and long sKK profiles.

Confirms in COMSOL what grazing_study.py established from TMM + the extracted PML law:
  1. how many elements a pf = 10, 20, 50 PML actually needs to reach its floor
     (predicted ~50*pf*cos(theta), i.e. ~43 elements at theta = 89 deg for pf = 50),
  2. the sKK grazing floor at L = 8, 32, 100 lambda, which is predicted to SATURATE.

Requires the Rice licence server (VPN).  ~250 solves.
Writes P9_* keys into master_data.npz, then:  python comsol_skk/make_figures.py 26
"""
import os, sys, time, numpy as np, jpype, mph
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); import tmm_helper as tmm_h
MODEL = os.path.join(ROOT, 'Comsol Files', 'skk_master.mph')
OUT   = os.path.join(HERE, 'master_data.npz')

lam = 0.8e-6; EB = 1.7**2
ANG = '0 45 80 85 89'; THS = np.array([0., 45., 80., 85., 89.])
NEL = sorted(set(int(round(v)) for v in np.geomspace(5, 2000, 24)))
PFS = [1, 4, 10, 20, 50]
LRS = [8, 32, 100]
RATIO = 80                      # self-similar family, w = L/80, ell = 2.5w (matches patch8)


def profile(kind, L, w):
    y = np.linspace(0.0, L, 6001)
    if kind == 'logistic':
        er = 1.0 + (EB - 1.0) / (1.0 + np.exp(-(y - L / 2) / w))
        return y, er, -tmm_h.ht_derivative(y, er), 1.0
    u = y - L / 2; D = u**2 + w**2; A = 4.0
    if kind == 'lorentzian':
        return y, EB + A * w * u / D, A * w**2 / D, EB
    if kind == 'tapered':
        return y, EB + A * w * u / D / np.cosh(u / (2.5 * w)), A * w**2 / D, EB
    raise ValueError(kind)


t0 = time.time(); client = mph.start(); model = client.load(MODEL); jm = model.java
comp = jm.component('comp1'); pml = comp.coordSystem('pml1'); STUDY = model.studies()[0]
old = dict(np.load(OUT)); RES = {}; nd = [0]
jm.study('std1').feature('stat').set('plistarr', jpype.JArray(jpype.JString)([ANG]))


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
        print(f'  !! {key}: {str(e).splitlines()[0][:60]}', flush=True); return None
    RES[key] = np.vstack([THS[:len(R)], R]); nd[0] += 1
    return R


# ---------------------------------------------------------------- A: high PMLfactor
print('\n=== A: PML element sweep at pf = 1, 4, 10, 20, 50 (PML length fixed 2 lam) ===',
      flush=True)
push(np.array([0.0, lam]), np.array([EB, EB]), np.array([0.0, 0.0]))
model.parameter('eps_low', 'n_bg^2')
model.parameter('L_term', '1*lam'); model.parameter('N_term', '400')
model.parameter('L_low', '2*lam'); pml.active(True)
for pf in PFS:
    pml.set('PMLfactor', str(pf))
    for N in NEL:
        model.parameter('N_low', str(N))
        run(f'P9_PML_pf{pf}_N{N}')
    v = RES.get(f'P9_PML_pf{pf}_N{NEL[-1]}')
    if v is not None:
        print(f'  pf={pf:2d}: ' + '  '.join(f'{t:.0f}deg={x:.3e}' for t, x in zip(THS, v[1])),
              flush=True)

# ---------------------------------------------------------------- B: long sKK profiles
print('\n=== B: sKK element sweep at L = 8, 32, 100 lam (w = L/80) ===', flush=True)
pml.active(False)
model.parameter('L_low', '2*lam'); model.parameter('N_low', '400')
for k in ['logistic', 'tapered']:
    for Lr in LRS:
        L = Lr * lam; w = L / RATIO
        y, er, ei, el = profile(k, L, w); push(y, er, ei)
        model.parameter('eps_low', repr(float(el)))
        model.parameter('L_term', f'{Lr}*lam')
        for N in NEL:
            model.parameter('N_term', str(N))
            run(f'P9_SKK_{k}_L{Lr}_N{N}')
        v = RES.get(f'P9_SKK_{k}_L{Lr}_N{NEL[-1]}')
        if v is not None:
            print(f'  {k:9s} L={Lr:3d}lam: '
                  + '  '.join(f'{t:.0f}deg={x:.3e}' for t, x in zip(THS, v[1])), flush=True)

old.update(RES); np.savez(OUT, **old); client.clear()
print(f'\nDONE {nd[0]} solves in {time.time()-t0:.0f}s -> {OUT}', flush=True)

# ---------------------------------------------------------------- summary
print('\n=== elements needed to reach the floor (within 10%) ===')
for key, lbl in ([(f'P9_PML_pf{pf}', f'PML pf={pf}') for pf in PFS]
                 + [(f'P9_SKK_{k}_L{Lr}', f'{k} L={Lr}lam')
                    for k in ['logistic', 'tapered'] for Lr in LRS]):
    have = [n for n in NEL if f'{key}_N{n}' in RES]
    if not have:
        continue
    for ti, t in enumerate(THS):
        vals = np.array([RES[f'{key}_N{n}'][1][ti] for n in have]); fin = vals[-1]
        n10 = next((n for n, x in zip(have, vals) if abs(x - fin) <= 0.1 * fin), None)
        print(f'  {lbl:22s} th={t:.0f}: floor={fin:.3e}  N_10% = {n10}')
