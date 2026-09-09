"""Tapered Lorentzian: eps' gets sech tails, eps'' kept as the Lorentzian (passive)."""
import os, sys, numpy as np, jpype, mph
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(ROOT, 'Comsol Files', 'skk_lorentzian_stacked.mph')
OUTM  = os.path.join(ROOT, 'Comsol Files', 'skk_lorentzian_tapered.mph')

client = mph.start(); model = client.load(MODEL); jm = model.java
comp = jm.component('comp1')
jm.param().set('ell', 'lam', 'sech taper length for eps_real')

Y0 = 'L_pml + L_lor/2'
DD = f'(((y-({Y0}))^2) + gam^2)'
RE = f'n_bg^2 - s_dir*A_lor*gam*(y-({Y0}))/{DD}/cosh((y-({Y0}))/ell)'
IM = f'A_lor*gam^2/{DD}'
wl = comp.physics('ewfd').feature('wee_lor')
wl.set('epsilonPrim', RE); wl.set('epsilonBis', IM)
print('eps_re =', RE); print('eps_im =', IM)

pml = comp.coordSystem('pml1'); STUDY = model.studies()[0]
model.parameter('L_lor','8*lam'); model.parameter('A_lor','4')
model.parameter('gam','lam/5'); model.parameter('s_dir','-1')
model.parameter('N_lor','960'); model.parameter('N_pml','400'); model.parameter('N_bg','40')

out = {}
def run(key, ell, pml_on):
    model.parameter('ell', ell); pml.active(bool(pml_on)); model.solve(STUDY)
    R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel(); th = np.linspace(0,89,len(R))
    out[key] = np.vstack([th, R])
    idx=[int(np.argmin(abs(th-t))) for t in [0,20,45,70,89]]
    print(f'{key:22s} R = ' + '  '.join(f'{R[i]:.3e}' for i in idx), flush=True)

print('\n--- tapered, PEC-backed (alone) ---')
for e in ['0.5*lam','1*lam','2*lam','4*lam']: run(f'alone_ell{e.split("*")[0]}', e, False)
run('alone_notaper', '1e6*lam', False)
print('\n--- tapered, stacked (+PML) ---')
for e in ['0.5*lam','1*lam','2*lam','4*lam']: run(f'stack_ell{e.split("*")[0]}', e, True)
run('stack_notaper', '1e6*lam', True)

np.savez(os.path.join(HERE,'lorentzian_tapered.npz'), **out)
model.save(OUTM); client.clear(); print('\nsaved', OUTM)
