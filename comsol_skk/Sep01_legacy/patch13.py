"""Mesh-grading comparison on ONE configuration: tapered Lorentzian, L = 10 lam.

Same model / port / solver as everything else in this folder; the ONLY thing that changes
is how the 50 elements across the terminator are distributed.  Three distributions, set on
mesh1/e_term/dis1:

    uniform       type = number                              (what every earlier run used)
    centre-dense  type = predefined, symmetric ON, reverse ON  -> fine in the middle
    entry-dense   type = predefined, reverse ON               -> fine at the entry face,
                                                                coarsening toward the PEC

Flag -> geometry mapping was measured on this model (40 elements, elemratio 20, in lam):
    predefined            dy = 0.031 (PEC) ... 0.620 (entry)
    predefined+symmetric  dy = 0.030 (both ends) ... 0.610 (centre)
    predefined+sym+rev    dy = 0.610 (both ends) ... 0.030 (centre)

N = 50 deliberately: at 800 elements over 10 lam (80 per lam) every distribution is
converged and the curves coincide.  A converged uniform N = 800 run is included as the
floor reference.  Writes P13_* keys.
"""
import os, sys, time, numpy as np, jpype, mph
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
MODEL = os.path.join(HERE, 'skk_master.mph')
OUT   = os.path.join(HERE, 'master_data.npz')

lam = 0.8e-6; EB = 1.7**2
FINE = 'range(0,1,89)'; TH = np.arange(0, 90, 1.0)
LR = 10.0; GAM = lam/5; ELL = 0.5*lam; AMP = 4.0     # tapered Lorentzian, fixed-width family
RATIOS = [5, 20, 100]


def tapered(L):
    y = np.linspace(0.0, L, 6001); u = y - L/2; D = u**2 + GAM**2
    return y, EB + AMP*GAM*u/D/np.cosh(u/ELL), AMP*GAM**2/D


t0 = time.time(); client = mph.start(); model = client.load(MODEL); jm = model.java
comp = jm.component('comp1'); pml = comp.coordSystem('pml1'); STUDY = model.studies()[0]
dis = comp.mesh('mesh1').feature('e_term').feature('dis1')
old = dict(np.load(OUT)); RES = {}
jm.study('std1').feature('stat').set('plistarr', jpype.JArray(jpype.JString)([FINE]))

y, er, ei = tapered(LR*lam)
comp.func('i_re').set('table', jpype.JArray(jpype.JString, 2)(
    [[repr(float(a)), repr(float(b))] for a, b in zip(y, er)]))
comp.func('i_im').set('table', jpype.JArray(jpype.JString, 2)(
    [[repr(float(a)), repr(float(b))] for a, b in zip(y, ei)]))
pml.active(False)
model.parameter('eps_low', repr(float(EB)))
model.parameter('L_term', f'{LR}*lam'); model.parameter('L_low', '2*lam')
model.parameter('N_low', '400')


def grading(kind, ratio=None):
    for k in ('symmetric', 'reverse'):
        dis.set(k, 'off')
    if kind == 'uniform':
        dis.set('type', 'number'); dis.set('numelem', 'N_term'); return
    dis.set('type', 'predefined'); dis.set('method', 'geometric')
    dis.set('elemcount', 'N_term'); dis.set('elemratio', str(ratio))
    dis.set('reverse', 'on')
    if kind == 'centre':
        dis.set('symmetric', 'on')


def run(key):
    try:
        model.solve(STUDY)
        R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
    except Exception as e:
        print(f'  !! {key}: {str(e).splitlines()[0][:70]}', flush=True); return
    RES[key] = np.vstack([TH[:len(R)], R])
    print(f'  {key:26s} R(0)={R[0]:.3e} R(45)={R[45]:.3e} R(80)={R[80]:.3e} '
          f'R(89)={R[89]:.3e}  [{time.time()-t0:.0f}s]', flush=True)


print('=== converged reference: uniform, N = 800 ===', flush=True)
model.parameter('N_term', '800'); grading('uniform'); run('P13_uniform_N800')

print('=== N = 50 ===', flush=True)
model.parameter('N_term', '50')
grading('uniform'); run('P13_uniform_N50')
for r in RATIOS:
    grading('centre', r); run(f'P13_ctr_r{r}_N50')
    grading('entry',  r); run(f'P13_ent_r{r}_N50')

grading('uniform')                      # leave the model as we found it
old.update(RES); np.savez(OUT, **old); client.clear()
print(f'DONE {len(RES)} solves in {time.time()-t0:.0f}s', flush=True)
