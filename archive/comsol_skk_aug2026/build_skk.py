"""Build the sKK Lorentzian terminator from the PML baseline, and scan A / orientation.

Profile (project convention, tmm_helper.eps):  eps(y) = n_b^2 - s*A*gam/((y-y0) + i*gam)
  Re eps = n_b^2 - s*A*gam*(y-y0)/D     Im eps = A*gam^2/D  > 0   (passive)
  D = (y-y0)^2 + gam^2
s = +1 / -1 selects which face is the reflectionless one.
"""
import os, numpy as np, mph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'Comsol Files', 'sKK_PML_baseline.mph')
OUTM = os.path.join(ROOT, 'Comsol Files', 'skk_lorentzian_v1.mph')
OUTD = os.path.dirname(os.path.abspath(__file__))

client = mph.start()
model = client.load(BASE)
jm = model.java
comp = jm.component('comp1')

# --- parameters for the sKK profile -----------------------------------------
jm.param().set('A_skk', '5',          'sKK Lorentzian amplitude')
jm.param().set('gam',   'L_test/20',  'sKK Lorentzian width')
jm.param().set('y0',    'L_test/2',   'sKK Lorentzian centre (y)')
jm.param().set('s_dir', '1',          'orientation: +1 or -1')

# --- turn the PML off --------------------------------------------------------
comp.coordSystem('pml1').active(False)

# --- make the background explicitly vacuum (wee1 was reading from a
#     RefractiveIndex group while set to DielectricLoss -- ambiguous) ---------
wee1 = comp.physics('ewfd').feature('wee1')
wee1.set('epsilonPrim_mat', 'userdef'); wee1.set('epsilonPrim', 'n_bg^2')
wee1.set('epsilonBis_mat',  'userdef'); wee1.set('epsilonBis',  '0')

# --- second Wave Equation node carrying the sKK profile on the terminator ----
D  = '((y-y0)^2 + gam^2)'
RE = f'n_bg^2 - s_dir*A_skk*gam*(y-y0)/{D}'
IM = f'A_skk*gam^2/{D}'
wee2 = comp.physics('ewfd').create('wee2', 'WaveEquationElectric', 2)
wee2.selection().set(1)                      # domain 1 = terminator
wee2.set('DisplacementFieldModel', 'DielectricLoss')
wee2.set('epsilonPrim_mat', 'userdef'); wee2.set('epsilonPrim', RE)
wee2.set('epsilonBis_mat',  'userdef'); wee2.set('epsilonBis',  IM)
wee2.set('mur_mat', 'userdef');         wee2.set('mur', '1')
wee2.set('sigma_mat', 'userdef');       wee2.set('sigma', '0')
wee2.label('sKK terminator')
print('wee2 created on domains', list(np.array(wee2.selection().entities()).ravel()))
print('  Re eps =', RE)
print('  Im eps =', IM)

model.save(OUTM)
print('saved', OUTM, flush=True)

# --- scan orientation and amplitude -----------------------------------------
STUDY = model.studies()[0]
angles_report = [0, 30, 60, 80, 89]
results = {}

for s in [1, -1]:
    for A in [1, 2, 5, 10]:
        model.parameter('s_dir', str(s))
        model.parameter('A_skk', str(A))
        model.solve(STUDY)
        R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
        Ab = np.asarray(model.evaluate('ewfd.Atotal')).ravel()
        th = np.linspace(0, 89, len(R))
        results[(s, A)] = (th, R, Ab)
        row = '  '.join(f'{R[int(np.argmin(abs(th-t)))]:.3e}' for t in angles_report)
        print(f's={s:+d} A={A:<3g} | R at {angles_report} = {row} | max|R+A-1|={np.max(abs(R+Ab-1)):.2e}',
              flush=True)

np.savez(os.path.join(OUTD, 'skk_scan_v1.npz'),
         **{f's{s}_A{A}': np.vstack(v[:2]) for (s, A), v in results.items()})
client.clear()
print('\nsaved -> comsol_skk/skk_scan_v1.npz')
