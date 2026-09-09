"""Stacked terminator: substrate | sKK profile (n_b -> air) | PML in air | PEC.

Fully parametric:
  eps'  = 1 + (n_bg^2-1)/(1+exp(-(y - L_pml - L_skk/2)/w))        analytic
  eps'' = (n_bg^2-1) * eps_im_u( (y-L_pml)/L_skk , w/L_skk )      2-D grid interpolation
DiHiTI is linear in eps' and scale invariant, so one unit-drop table covers all
n_bg, L_skk and w.
"""
import os, sys, numpy as np, jpype, mph
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); import tmm_helper as tmm_h
HERE = os.path.dirname(os.path.abspath(__file__))
OUTM = os.path.join(ROOT, 'Comsol Files', 'skk_stacked_v1.mph')
GRID = os.path.join(ROOT, 'Comsol Files', 'skk_eps_im_unit.txt')

# ---------- 1. build the unit-drop eps'' table on (xi, r=w/L) ----------------
XI = np.linspace(0.0, 1.0, 401)
RR = np.geomspace(0.005, 0.10, 25)          # w/L from 1/200 to 1/10
tab = np.empty((len(RR), len(XI)))
for j, r in enumerate(RR):
    er = 1.0/(1.0 + np.exp(-(XI - 0.5)/r))   # unit drop: 0 at xi=0, 1 at xi=1
    tab[j] = -tmm_h.ht_derivative(XI, er)
assert tab.min() > -1e-9, f'unit table not passive: {tab.min()}'
with open(GRID, 'w') as fh:
    fh.write('% Grid\n')
    fh.write(' '.join(f'{v:.10g}' for v in XI) + '\n')
    fh.write(' '.join(f'{v:.10g}' for v in RR) + '\n')
    fh.write('% Data\n')
    for row in tab:
        fh.write(' '.join(f'{v:.10g}' for v in row) + '\n')
print(f'grid file: {len(XI)} xi x {len(RR)} r, eps_im range [{tab.min():.3f}, {tab.max():.3f}]')

# ---------- 2. build the model from scratch ---------------------------------
client = mph.start()
model = client.create('skk_stacked')
jm = model.java
P = [('lam','800[nm]','wavelength'), ('f0','c_const/lam','frequency'),
     ('theta','0[deg]','angle of incidence in substrate'), ('n_bg','1.7','substrate index'),
     ('Lx','lam/4','period'), ('L_bg','lam','substrate region'),
     ('L_skk','6*lam','sKK profile thickness'), ('L_pml','lam','PML thickness (in air)'),
     ('w','L_skk/40','logistic feature width'),
     ('N_skk','400','elements in sKK'), ('N_pml','100','elements in PML'),
     ('N_bg','40','elements in substrate'), ('N_x','5','elements across period')]
for k, v, d in P: jm.param().set(k, v, d)

comp = jm.component().create('comp1', True)
geom = comp.geom().create('geom1', 2)
r1 = geom.create('r1', 'Rectangle')
r1.set('size', jpype.JArray(jpype.JString)(['Lx', 'L_pml+L_skk+L_bg']))
r1.set('layername', jpype.JArray(jpype.JString)(['pml', 'skk']))
r1.setIndex('layer', 'L_pml', 0); r1.setIndex('layer', 'L_skk', 1)
geom.run()

# ---------- 3. identify domains and boundaries by geometry -------------------
nb_ = int(geom.getNBoundaries()); nd_ = int(geom.getNDomains())
vc = np.array(geom.getVertexCoord())
adj = [list(np.array(r).ravel()) for r in geom.getAdj(1, 0)]
mid = {}
for b in range(1, nb_+1):
    vs = [int(v)-1 for v in adj[b]]
    mid[b] = vc[:, vs].mean(axis=1)
ud = np.array(geom.getUpDown())
xs = sorted({round(m[0], 12) for m in mid.values()})
x_lo, x_hi = xs[0], xs[-1]
verticals = [b for b, m in mid.items() if abs(m[0]-x_lo) < 1e-12 or abs(m[0]-x_hi) < 1e-12]
horizontals = [b for b in mid if b not in verticals]
b_top = max(horizontals, key=lambda b: mid[b][1])
b_bot = min(horizontals, key=lambda b: mid[b][1])
# domain centroids from the y of their adjacent vertical boundaries
dom_y = {}
for b in verticals:
    for d in (int(ud[0][b-1]), int(ud[1][b-1])):
        if d > 0: dom_y.setdefault(d, []).append(mid[b][1])
order = sorted(dom_y, key=lambda d: np.mean(dom_y[d]))
d_pml, d_skk, d_bg = order            # bottom -> top
print(f'{nd_} domains, {nb_} boundaries')
print(f'  domains bottom->top: PML={d_pml}, sKK={d_skk}, substrate={d_bg}')
print(f'  port boundary={b_top} (y={mid[b_top][1]*1e6:.3f} um), PEC={b_bot}, verticals={verticals}')

# ---------- 4. functions ----------------------------------------------------
fi = comp.func().create('int_im', 'Interpolation')
fi.set('source', 'file'); fi.set('filename', GRID)
fi.setIndex('funcs', 'eps_im_u', 0, 0)
fi.set('interp', 'linear'); fi.set('extrap', 'const')
try:
    fi.set('argunit', jpype.JArray(jpype.JString)(['1', '1']))
except Exception:
    fi.set('argunit', '1,1')
fi.set('fununit', '1')

# ---------- 5. materials, PML, physics --------------------------------------
mat = comp.material().create('mat1', 'Common'); mat.label('background')
mat.propertyGroup('def').set('relpermeability', jpype.JArray(jpype.JString)(['1','0','0','0','1','0','0','0','1']))
mat.propertyGroup('def').set('electricconductivity', jpype.JArray(jpype.JString)(['0']*9))

pml = comp.coordSystem().create('pml1', 'PML')
pml.selection().set(jpype.JArray(jpype.JInt)([d_pml]))
pml.set('PMLfactor', '1')

ph = comp.physics().create('ewfd', 'ElectromagneticWavesFrequencyDomain', 'geom1')
ph.prop('EquationForm').set('form', 'Frequency')
ph.prop('EquationForm').set('freq_src', 'userdef')
ph.prop('EquationForm').set('freq', 'f0')
ph.prop('components').set('components', 'outofplane')
w1 = ph.feature('wee1')
w1.set('DisplacementFieldModel', 'DielectricLoss')
w1.set('minput_frequency_src','userdef'); w1.set('minput_frequency','f0')
w1.set('epsilonPrim_mat','userdef'); w1.set('epsilonPrim','n_bg^2')
w1.set('epsilonBis_mat','userdef');  w1.set('epsilonBis','0')

# air inside the PML region
w_air = ph.create('wee_air', 'WaveEquationElectric', 2)
w_air.selection().set(jpype.JArray(jpype.JInt)([d_pml])); w_air.label('air (PML region)')
w_air.set('DisplacementFieldModel','DielectricLoss')
w_air.set('epsilonPrim_mat','userdef'); w_air.set('epsilonPrim','1')
w_air.set('epsilonBis_mat','userdef');  w_air.set('epsilonBis','0')
w_air.set('mur_mat','userdef'); w_air.set('mur','1')
w_air.set('sigma_mat','userdef'); w_air.set('sigma','0')

# sKK graded profile
EPR = '1 + (n_bg^2-1)/(1+exp(-(y - L_pml - L_skk/2)/w))'
EPI = '(n_bg^2-1)*eps_im_u((y-L_pml)/L_skk, w/L_skk)'
w_skk = ph.create('wee_skk', 'WaveEquationElectric', 2)
w_skk.selection().set(jpype.JArray(jpype.JInt)([d_skk])); w_skk.label('sKK profile')
w_skk.set('DisplacementFieldModel','DielectricLoss')
w_skk.set('epsilonPrim_mat','userdef'); w_skk.set('epsilonPrim', EPR)
w_skk.set('epsilonBis_mat','userdef');  w_skk.set('epsilonBis', EPI)
w_skk.set('mur_mat','userdef'); w_skk.set('mur','1')
w_skk.set('sigma_mat','userdef'); w_skk.set('sigma','0')

pc = ph.create('pc1', 'PeriodicCondition', 1)
pc.selection().set(jpype.JArray(jpype.JInt)(verticals))
pc.set('PeriodicType', 'Floquet'); pc.set('Floquet_source', 'FromPeriodicPort')
pt = ph.create('port1', 'Port', 1)
pt.selection().set(jpype.JArray(jpype.JInt)([b_top]))
pt.set('PortType', 'Periodic'); pt.set('alpha_inc', 'theta')
pt.set('E0', jpype.JArray(jpype.JInt, 2)([[0],[0],[1]]))
pt.set('Eampl', jpype.JArray(jpype.JInt, 2)([[0],[0],[1]]))

# ---------- 6. mesh ---------------------------------------------------------
msh = comp.mesh().create('mesh1', 'geom1')
def edge_dist(tag, bnd, n):
    e = msh.create(tag, 'Edge'); e.selection().set(jpype.JArray(jpype.JInt)([bnd]))
    e.create('dis1', 'Distribution').set('numelem', n)
left = sorted([b for b in verticals if abs(mid[b][0]-x_lo) < 1e-12], key=lambda b: mid[b][1])
edge_dist('e_pml', left[0], 'N_pml'); edge_dist('e_skk', left[1], 'N_skk'); edge_dist('e_bg', left[2], 'N_bg')
edge_dist('e_x', b_bot, 'N_x')
for tag, src in [('cp1', left[0]), ('cp2', left[1]), ('cp3', left[2])]:
    c = msh.create(tag, 'CopyEdge')
    c.selection('source').set(jpype.JArray(jpype.JInt)([src]))
    rt = [b for b in verticals if abs(mid[b][0]-x_hi) < 1e-12 and abs(mid[b][1]-mid[src][1]) < 1e-12]
    c.selection('destination').set(jpype.JArray(jpype.JInt)(rt))
msh.create('map1', 'Map'); msh.run()
print('mesh built')

std = jm.study().create('std1'); std.create('stat', 'Stationary')
std.feature('stat').set('useparam', True)
std.feature('stat').set('pname', jpype.JArray(jpype.JString)(['theta']))
std.feature('stat').set('plistarr', jpype.JArray(jpype.JString)(['range(0,1,89)']))
std.feature('stat').set('punit', jpype.JArray(jpype.JString)(['deg']))
jm.sol().create('sol1').attach('std1')
jm.sol('sol1').createAutoSequence('std1')
model.save(OUTM); print('saved', OUTM, flush=True)

# ---------- 7. solve --------------------------------------------------------
STUDY = model.studies()[0]; out = {}
for mult in [2, 4, 6]:
    model.parameter('L_skk', f'{mult}*lam')
    model.solve(STUDY)
    R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel(); th = np.linspace(0, 89, len(R))
    out[f'stacked_L{mult}'] = np.vstack([th, R])
    idx = [int(np.argmin(abs(th-t))) for t in [0, 20, 35, 60, 89]]
    print(f'stacked L_skk={mult}lam  R = ' + '  '.join(f'{R[i]:.2e}' for i in idx), flush=True)

# verify the interpolation reproduces the Python profile
model.parameter('L_skk', '6*lam')
model.solve(STUDY)
yv, ev = model.evaluate(['y', EPI])
yv = np.asarray(yv).ravel(); ev = np.asarray(ev).ravel()
Lp, Ls = 0.8e-6, 6*0.8e-6
m = (yv > Lp+1e-12) & (yv < Lp+Ls-1e-12)
xi = (yv[m]-Lp)/Ls
ref = np.interp(xi, XI, (1.7**2-1)*tab[np.argmin(abs(RR-1/40))])
print(f'\ninterpolation check: max |COMSOL - python| = {np.max(np.abs(ev[m]-ref)):.3e} '
      f'(eps_im peak {ref.max():.3f})')

np.savez(os.path.join(HERE, 'skk_stacked_v1.npz'), **out)
model.save(OUTM); client.clear(); print('done')
