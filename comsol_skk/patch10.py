"""Fine angle sweep at CONVERGED mesh, for every terminator in patch9.

patch9 measured on a 5-angle grid and found the logistic at L=100 lam dipping below the
PML floor at 85 deg.  This maps that window at 1 deg resolution so we know whether it is
a real angular window or an artefact of the coarse grid.  Same model, same mesh rule.
Writes P10_* keys.  ~11 solves.
"""
import os, sys, time, numpy as np, jpype, mph
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); import tmm_helper as tmm_h
MODEL = os.path.join(ROOT, 'Comsol Files', 'skk_master.mph')
OUT   = os.path.join(HERE, 'master_data.npz')

lam = 0.8e-6; EB = 1.7**2
FINE = 'range(0,1,89)'; TH = np.arange(0, 90, 1.0)
NCONV = 2000                      # every configuration in patch9 was converged by 2000

def profile(kind, L, w):
    y = np.linspace(0.0, L, 6001)
    if kind == 'logistic':
        er = 1.0 + (EB - 1.0) / (1.0 + np.exp(-(y - L / 2) / w))
        return y, er, -tmm_h.ht_derivative(y, er), 1.0
    u = y - L / 2; D = u**2 + w**2; A = 4.0
    if kind == 'tapered':
        return y, EB + A * w * u / D / np.cosh(u / (2.5 * w)), A * w**2 / D, EB
    raise ValueError(kind)

t0 = time.time(); client = mph.start(); model = client.load(MODEL); jm = model.java
comp = jm.component('comp1'); pml = comp.coordSystem('pml1'); STUDY = model.studies()[0]
old = dict(np.load(OUT)); RES = {}
jm.study('std1').feature('stat').set('plistarr', jpype.JArray(jpype.JString)([FINE]))

def push(y, er, ei):
    comp.func('i_re').set('table', jpype.JArray(jpype.JString, 2)(
        [[repr(float(a)), repr(float(b))] for a, b in zip(y, er)]))
    comp.func('i_im').set('table', jpype.JArray(jpype.JString, 2)(
        [[repr(float(a)), repr(float(b))] for a, b in zip(y, ei)]))

def run(key):
    model.solve(STUDY)
    R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
    RES[key] = np.vstack([TH[:len(R)], R])
    print(f'  {key:28s} R(80)={R[80]:.3e} R(85)={R[85]:.3e} R(89)={R[89]:.3e}', flush=True)

print('=== PML, fine angle, converged mesh ===', flush=True)
push(np.array([0.0, lam]), np.array([EB, EB]), np.array([0.0, 0.0]))
model.parameter('eps_low', 'n_bg^2')
model.parameter('L_term', '1*lam'); model.parameter('N_term', '400')
model.parameter('L_low', '2*lam'); model.parameter('N_low', str(NCONV)); pml.active(True)
for pf in [1, 4, 10, 20, 50]:
    pml.set('PMLfactor', str(pf)); run(f'P10_PML_pf{pf}')

print('=== sKK, fine angle, converged mesh ===', flush=True)
pml.active(False)
model.parameter('L_low', '2*lam'); model.parameter('N_low', '400')
model.parameter('N_term', str(NCONV))
for k in ['logistic', 'tapered']:
    for Lr in [8, 32, 100]:
        L = Lr * lam; y, er, ei, el = profile(k, L, L / 80); push(y, er, ei)
        model.parameter('eps_low', repr(float(el)))
        model.parameter('L_term', f'{Lr}*lam')
        run(f'P10_SKK_{k}_L{Lr}')

old.update(RES); np.savez(OUT, **old); client.clear()
print(f'DONE in {time.time()-t0:.0f}s', flush=True)
