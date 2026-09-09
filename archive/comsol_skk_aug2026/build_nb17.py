"""Paper configuration: background n_b=1.7, logistic sKK terminating in air.
eps' = n_b^2 at the entry face, dropping to 1 at the PEC end; ht sign = -1 (passive branch)."""
import os, sys, numpy as np, jpype, mph
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tmm_helper as tmm_h
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, 'Comsol Files', 'sKK_PML_baseline.mph')
OUTM = os.path.join(ROOT, 'Comsol Files', 'skk_nb17_v1.mph')

lam = 0.8e-6; nb = 1.7; eb = nb**2
L_test = 2*lam; y0 = L_test/2
y = np.linspace(0.0, L_test, 2001)

def profile(w, skk=True):
    if not skk:
        return np.full_like(y, eb), np.zeros_like(y)
    er = 1.0 + (eb - 1.0)/(1.0 + np.exp(-(y - y0)/w))   # eb at y=L (entry) -> 1 at y=0 (PEC)
    ei = -tmm_h.ht_derivative(y, er)                     # passive branch
    assert ei.min() > -1e-9, ei.min()
    return er, ei

def tbl(vs):
    return jpype.JArray(jpype.JString, 2)([[repr(float(a)), repr(float(b))] for a, b in zip(y, vs)])

client = mph.start(); model = client.load(BASE)
jm = model.java; comp = jm.component('comp1')
model.parameter('n_bg', str(nb))
wee1 = comp.physics('ewfd').feature('wee1')
wee1.set('epsilonPrim_mat','userdef'); wee1.set('epsilonPrim','n_bg^2')
wee1.set('epsilonBis_mat','userdef');  wee1.set('epsilonBis','0')

er0, ei0 = profile(L_test/40)
for tag, fn, vs in [('int_re','eps_re',er0), ('int_im','eps_im',ei0)]:
    f = comp.func().create(tag,'Interpolation')
    f.set('source','table'); f.set('table', tbl(vs)); f.set('funcname', fn)
    f.set('interp','linear'); f.set('extrap','const'); f.set('argunit','m'); f.set('fununit','1')

wee2 = comp.physics('ewfd').create('wee2','WaveEquationElectric',2)
wee2.selection().set(1)
wee2.set('DisplacementFieldModel','DielectricLoss')
wee2.set('epsilonPrim_mat','userdef'); wee2.set('epsilonPrim','eps_re(y)')
wee2.set('epsilonBis_mat','userdef');  wee2.set('epsilonBis','eps_im(y)')
wee2.set('mur_mat','userdef');   wee2.set('mur','1')
wee2.set('sigma_mat','userdef'); wee2.set('sigma','0')
wee2.label('sKK logistic terminator')

STUDY = model.studies()[0]; pml = comp.coordSystem('pml1'); out = {}
theta_c = np.degrees(np.arcsin(1.0/nb))
print(f'n_b = {nb}, eps_b = {eb}, TIR angle = {theta_c:.1f} deg\n')

configs = [('PML_pf1',  True, 1, None), ('PML_pf4',  True, 4, None),
           ('sKK_w_L20', False, 1, L_test/20),
           ('sKK_w_L40', False, 1, L_test/40),
           ('sKK_w_L80', False, 1, L_test/80)]
for name, pml_on, pf, w in configs:
    pml.active(bool(pml_on))
    if pml_on: pml.set('PMLfactor', str(pf))
    er, ei = profile(w if w else L_test/40, skk=not pml_on)
    comp.func('int_re').set('table', tbl(er)); comp.func('int_im').set('table', tbl(ei))
    # when the PML is on, the terminator must be the plain background medium
    wee2.active(not pml_on)
    model.solve(STUDY)
    R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
    th = np.linspace(0, 89, len(R)); out[name] = np.vstack([th, R])
    idx = [int(np.argmin(abs(th-t))) for t in [0,10,20,30,35,60,89]]
    print(f'{name:11s} R at [0,10,20,30,35,60,89] = ' + '  '.join(f'{R[i]:.2e}' for i in idx), flush=True)

np.savez(os.path.join(HERE,'skk_nb17_v1.npz'), theta_c=theta_c, **out)
np.savez(os.path.join(HERE,'profile_nb17.npz'), y=y, eps_re=er0, eps_im=ei0)
model.save(OUTM); client.clear(); print('\nsaved', OUTM)
