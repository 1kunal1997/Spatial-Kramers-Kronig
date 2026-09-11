"""Extend the L=100 lam equal-cost sweep to N=8000, and add the plain Lorentzian.

Reuses patch9's geometry EXACTLY so the new points concatenate onto the existing
P9_* element sweep:
    sKK : L_term = 100 lam, N_term = N, L_low = 2 lam, N_low = 400
    PML : L_term = 1 lam,   N_term = 400, L_low = 2 lam, N_low = N
Self-similar profile family w = L/80 = 1.25 lam, ell = 2.5w  (the fig-19 recipe),
so these panels tie to the terminator-length slide, not to the L=8 lam slide.

The element grid continues patch9's geometric progression (ratio (2000/5)^(1/23))
past 2000, so the low-N sampling is identical to the other equal-cost plots.
Writes P9_* keys for the new N values, plus the Lorentzian over the full grid.
"""
import os, sys, time, numpy as np, jpype, mph
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); import tmm_helper as tmm_h
MODEL = os.path.join(HERE, 'skk_master.mph')
OUT   = os.path.join(HERE, 'master_data.npz')

lam = 0.8e-6; EB = 1.7**2
ANG = '0 45 80 85 89'; THS = np.array([0., 45., 80., 85., 89.])
NEL = sorted(set(int(round(v)) for v in np.geomspace(5, 2000, 24)))
r   = (2000 / 5) ** (1 / 23)
EXT = sorted(set([int(round(2000 * r**k)) for k in range(1, 6)] + [8000]))
LR, RATIO = 100, 80
print('existing grid tops out at', NEL[-1], '-> appending', EXT, flush=True)


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
        print(f'  !! {key}: {str(e).splitlines()[0][:60]}', flush=True); return
    RES[key] = np.vstack([THS[:len(R)], R]); nd[0] += 1


print('\n=== sKK at L=100 lam, w=L/80 ===', flush=True)
pml.active(False)
model.parameter('L_low', '2*lam'); model.parameter('N_low', '400')
model.parameter('L_term', f'{LR}*lam')
L = LR * lam; w = L / RATIO
for k in ['lorentzian', 'logistic', 'tapered']:
    todo = (NEL + EXT) if k == 'lorentzian' else EXT      # the other two already have 5..2000
    y, er, ei, el = profile(k, L, w); push(y, er, ei)
    model.parameter('eps_low', repr(float(el)))
    for N in todo:
        model.parameter('N_term', str(N)); run(f'P9_SKK_{k}_L{LR}_N{N}')
    v = RES.get(f'P9_SKK_{k}_L{LR}_N{EXT[-1]}')
    if v is not None:
        print(f'  {k:11s} N={EXT[-1]}: ' + '  '.join(f'{t:.0f}deg={x:.3e}'
              for t, x in zip(THS, v[1])), flush=True)

print('\n=== PML pf=1,4 extended ===', flush=True)
push(np.array([0.0, lam]), np.array([EB, EB]), np.array([0.0, 0.0]))
model.parameter('eps_low', 'n_bg^2')
model.parameter('L_term', '1*lam'); model.parameter('N_term', '400')
model.parameter('L_low', '2*lam'); pml.active(True)
for pf in [1, 4]:
    pml.set('PMLfactor', str(pf))
    for N in EXT:
        model.parameter('N_low', str(N)); run(f'P9_PML_pf{pf}_N{N}')
    v = RES.get(f'P9_PML_pf{pf}_N{EXT[-1]}')
    if v is not None:
        print(f'  pf={pf} N={EXT[-1]}: ' + '  '.join(f'{t:.0f}deg={x:.3e}'
              for t, x in zip(THS, v[1])), flush=True)

old.update(RES); np.savez(OUT, **old); client.clear()
print(f'\nDONE {nd[0]} solves in {time.time()-t0:.0f}s', flush=True)
