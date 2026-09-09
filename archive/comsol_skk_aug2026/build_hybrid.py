"""Hybrid terminator: PML coordinate stretch AND the logistic sKK material together.
D=0 reduces to the pure PML, so that config is a built-in control."""
import os, sys, numpy as np, jpype, mph
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tmm_helper as tmm_h
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, 'Comsol Files', 'sKK_PML_baseline.mph')
OUTM = os.path.join(ROOT, 'Comsol Files', 'skk_hybrid_v1.mph')

lam = 0.8e-6; L_test = 2*lam; y0 = L_test/2; w = L_test/40
y = np.linspace(0.0, L_test, 2001)

def profile(D):
    if D == 0:
        return np.ones_like(y), np.zeros_like(y)
    er = 1.0 + D/(1.0 + np.exp((y - y0)/w))
    ei = tmm_h.ht_derivative(y, er)
    assert ei.min() > -1e-9
    return er, ei

def tbl(vs):
    return jpype.JArray(jpype.JString, 2)(
        [[repr(float(a)), repr(float(b))] for a, b in zip(y, vs)])

client = mph.start(); model = client.load(BASE)
jm = model.java; comp = jm.component('comp1')
wee1 = comp.physics('ewfd').feature('wee1')
wee1.set('epsilonPrim_mat', 'userdef'); wee1.set('epsilonPrim', 'n_bg^2')
wee1.set('epsilonBis_mat',  'userdef'); wee1.set('epsilonBis',  '0')

for tag, fname, vs in [('int_re', 'eps_re', profile(3.6)[0]), ('int_im', 'eps_im', profile(3.6)[1])]:
    f = comp.func().create(tag, 'Interpolation')
    f.set('source', 'table'); f.set('table', tbl(vs))
    f.set('funcname', fname); f.set('interp', 'linear')
    f.set('extrap', 'const'); f.set('argunit', 'm'); f.set('fununit', '1')

wee2 = comp.physics('ewfd').create('wee2', 'WaveEquationElectric', 2)
wee2.selection().set(1)
wee2.set('DisplacementFieldModel', 'DielectricLoss')
wee2.set('epsilonPrim_mat', 'userdef'); wee2.set('epsilonPrim', 'eps_re(y)')
wee2.set('epsilonBis_mat',  'userdef'); wee2.set('epsilonBis',  'eps_im(y)')
wee2.set('mur_mat', 'userdef');   wee2.set('mur', '1')
wee2.set('sigma_mat', 'userdef'); wee2.set('sigma', '0')
wee2.label('sKK logistic terminator')

STUDY = model.studies()[0]
pml = comp.coordSystem('pml1')
out = {}

configs = [(True, 1, 0), (True, 1, 1), (True, 1, 3.6), (True, 1, 8),
           (True, 4, 3.6), (False, 1, 3.6)]
for pml_on, pf, D in configs:
    pml.active(bool(pml_on)); pml.set('PMLfactor', str(pf))
    er, ei = profile(D)
    comp.func('int_re').set('table', tbl(er))
    comp.func('int_im').set('table', tbl(ei))
    model.solve(STUDY)
    R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
    th = np.linspace(0, 89, len(R))
    key = f'{"PML" if pml_on else "noPML"}_pf{pf}_D{D}'
    out[key] = np.vstack([th, R])
    idx = [int(np.argmin(abs(th - t))) for t in [0, 30, 60, 80, 89]]
    print(f'{key:22s} R = ' + '  '.join(f'{R[i]:.3e}' for i in idx), flush=True)

np.savez(os.path.join(HERE, 'skk_hybrid_v1.npz'), **out)
model.save(OUTM); client.clear()
print('saved', OUTM)
