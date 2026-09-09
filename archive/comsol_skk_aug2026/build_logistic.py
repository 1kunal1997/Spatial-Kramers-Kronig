"""Logistic (DiHiTI) sKK terminator: eps' monotone logistic, eps'' from ht_derivative,
embedded in COMSOL as interpolation functions of y."""
import os, sys, numpy as np, jpype, mph
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tmm_helper as tmm_h

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, 'Comsol Files', 'sKK_PML_baseline.mph')
OUTM = os.path.join(ROOT, 'Comsol Files', 'skk_logistic_v1.mph')

lam = 0.8e-6; L_test = 2*lam; y0 = L_test/2; w = L_test/40
NPTS = 2001
y = np.linspace(0.0, L_test, NPTS)

def profile(D):
    er = 1.0 + D/(1.0 + np.exp((y - y0)/w))      # 1 at entry (y=L), 1+D at PEC (y=0)
    ei = tmm_h.ht_derivative(y, er)
    assert ei.min() > -1e-9, f'not passive: min eps_im = {ei.min()}'
    return er, ei

client = mph.start()
model = client.load(BASE)
jm = model.java; comp = jm.component('comp1')
comp.coordSystem('pml1').active(False)
wee1 = comp.physics('ewfd').feature('wee1')
wee1.set('epsilonPrim_mat', 'userdef'); wee1.set('epsilonPrim', 'n_bg^2')
wee1.set('epsilonBis_mat',  'userdef'); wee1.set('epsilonBis',  '0')

def make_interp(tag, fname, xs, vs):
    f = comp.func().create(tag, 'Interpolation')
    f.set('source', 'table')
    tbl = jpype.JArray(jpype.JString, 2)([[repr(float(a)), repr(float(b))] for a, b in zip(xs, vs)])
    f.set('table', tbl)
    for prop, val in [('funcname', fname), ('interp', 'linear'),
                      ('extrap', 'const'), ('argunit', 'm'), ('fununit', '1')]:
        try: f.set(prop, val)
        except Exception as e: print(f'  warn {tag}.{prop}: {e}')
    return f

er, ei = profile(3.6)
make_interp('int_re', 'eps_re', y, er)
make_interp('int_im', 'eps_im', y, ei)
print('interpolation functions created')

wee2 = comp.physics('ewfd').create('wee2', 'WaveEquationElectric', 2)
wee2.selection().set(1)
wee2.set('DisplacementFieldModel', 'DielectricLoss')
wee2.set('epsilonPrim_mat', 'userdef'); wee2.set('epsilonPrim', 'eps_re(y)')
wee2.set('epsilonBis_mat',  'userdef'); wee2.set('epsilonBis',  'eps_im(y)')
wee2.set('mur_mat', 'userdef');   wee2.set('mur', '1')
wee2.set('sigma_mat', 'userdef'); wee2.set('sigma', '0')
wee2.label('sKK logistic terminator')

STUDY = model.studies()[0]
out = {}
for D in [1, 2, 3.6, 5, 8]:
    er, ei = profile(D)
    # rewrite both tables for this D
    comp.func('int_re').set('table',
        jpype.JArray(jpype.JString, 2)([[repr(float(a)), repr(float(b))] for a, b in zip(y, er)]))
    comp.func('int_im').set('table',
        jpype.JArray(jpype.JString, 2)([[repr(float(a)), repr(float(b))] for a, b in zip(y, ei)]))
    model.solve(STUDY)
    R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
    th = np.linspace(0, 89, len(R))
    out[f'D{D}'] = np.vstack([th, R])
    idx = [int(np.argmin(abs(th - t))) for t in [0, 30, 60, 80, 89]]
    print(f'D={D:<4g} | R = ' + '  '.join(f'{R[i]:.3e}' for i in idx) +
          f'  | min {R.min():.3e} @ {th[int(np.argmin(R))]:.0f} deg', flush=True)

np.savez(os.path.join(HERE, 'skk_logistic_v1.npz'), **out)
np.savez(os.path.join(HERE, 'logistic_profile_D3p6.npz'), y=y, eps_re=profile(3.6)[0], eps_im=profile(3.6)[1])
model.save(OUTM); client.clear()
print('saved', OUTM)
