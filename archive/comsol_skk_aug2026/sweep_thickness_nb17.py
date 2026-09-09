"""n_b=1.7 sKK terminating in air: sweep terminator thickness (w scales as L/40)."""
import os, sys, numpy as np, jpype, mph
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); import tmm_helper as tmm_h
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, 'Comsol Files', 'sKK_PML_baseline.mph')

lam = 0.8e-6; nb = 1.7; eb = nb**2; NP = 3001

def profile(Lt, skk=True):
    y = np.linspace(0.0, Lt, NP)
    if not skk: return y, np.full_like(y, eb), np.zeros_like(y)
    w = Lt/40; y0 = Lt/2
    er = 1.0 + (eb-1.0)/(1.0 + np.exp(-(y-y0)/w))
    ei = -tmm_h.ht_derivative(y, er)
    assert ei.min() > -1e-9
    return y, er, ei

def tbl(y, vs):
    return jpype.JArray(jpype.JString, 2)([[repr(float(a)), repr(float(b))] for a, b in zip(y, vs)])

client = mph.start(); model = client.load(BASE)
jm = model.java; comp = jm.component('comp1')
model.parameter('n_bg', str(nb)); model.parameter('N_elem', '400')
w1 = comp.physics('ewfd').feature('wee1')
w1.set('epsilonPrim_mat','userdef'); w1.set('epsilonPrim','n_bg^2')
w1.set('epsilonBis_mat','userdef');  w1.set('epsilonBis','0')
y0_, er0, ei0 = profile(2*lam)
for tag, fn, vs in [('int_re','eps_re',er0), ('int_im','eps_im',ei0)]:
    f = comp.func().create(tag,'Interpolation'); f.set('source','table')
    f.set('table', tbl(y0_, vs)); f.set('funcname', fn)
    f.set('interp','linear'); f.set('extrap','const'); f.set('argunit','m'); f.set('fununit','1')
wee2 = comp.physics('ewfd').create('wee2','WaveEquationElectric',2); wee2.selection().set(1)
wee2.set('DisplacementFieldModel','DielectricLoss')
wee2.set('epsilonPrim_mat','userdef'); wee2.set('epsilonPrim','eps_re(y)')
wee2.set('epsilonBis_mat','userdef');  wee2.set('epsilonBis','eps_im(y)')
wee2.set('mur_mat','userdef'); wee2.set('mur','1')
wee2.set('sigma_mat','userdef'); wee2.set('sigma','0')

STUDY = model.studies()[0]; pml = comp.coordSystem('pml1'); out = {}
k0 = 2*np.pi/lam
for mult in [2, 4, 6, 8, 12]:
    Lt = mult*lam
    model.parameter('L_test', f'{mult}*lam')
    pml.active(False); wee2.active(True)
    y, er, ei = profile(Lt)
    comp.func('int_re').set('table', tbl(y, er)); comp.func('int_im').set('table', tbl(y, ei))
    S = np.trapezoid(ei/np.sqrt(np.maximum(er,1e-9)), y)
    model.solve(STUDY)
    R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel(); th = np.linspace(0,89,len(R))
    out[f'sKK_L{mult}'] = np.vstack([th, R])
    idx=[int(np.argmin(abs(th-t))) for t in [0,20,35,60,89]]
    print(f'sKK L={mult:>2}lam  leak_pred={np.exp(-2*k0*S):.2e}  R = '
          + '  '.join(f'{R[i]:.2e}' for i in idx), flush=True)

for pf in [1, 4]:
    model.parameter('L_test', '2*lam')
    pml.active(True); pml.set('PMLfactor', str(pf)); wee2.active(False)
    model.solve(STUDY)
    R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel(); th = np.linspace(0,89,len(R))
    out[f'PML_pf{pf}'] = np.vstack([th, R])
    idx=[int(np.argmin(abs(th-t))) for t in [0,20,35,60,89]]
    print(f'PML pf={pf} L=2lam                    R = ' + '  '.join(f'{R[i]:.2e}' for i in idx), flush=True)

np.savez(os.path.join(HERE,'skk_nb17_thickness.npz'), **out)
client.clear(); print('done')
