"""Lorentzian sKK terminator, eps_b on BOTH sides (no index step, no TIR, honest 0-90 deg).

    eps(y) = eps_b - s*A*gam/((y-y0) + i*gam)
    Re = eps_b - s*A*gam*(y-y0)/D      Im = A*gam^2/D > 0     D=(y-y0)^2+gam^2

Geometry: substrate(eps_b) | Lorentzian region L_lor | lower region L_pml | PEC
PML on  -> "stacked";  PML off -> "alone" (lossless gap then PEC).
"""
import os, sys, numpy as np, jpype, mph
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
OUTM = os.path.join(ROOT, 'Comsol Files', 'skk_lorentzian_stacked.mph')

client = mph.start(); model = client.create('lor_stacked'); jm = model.java
for k, v, d in [('lam','800[nm]','wavelength'), ('f0','c_const/lam','frequency'),
    ('theta','0[deg]','angle in background'), ('n_bg','1.7','background index'),
    ('Lx','lam/4','period'), ('L_bg','lam','substrate region'),
    ('L_lor','8*lam','Lorentzian region (truncation length)'),
    ('L_pml','lam','lower region / PML'),
    ('A_lor','4','Lorentzian amplitude'), ('gam','lam/5','Lorentzian width'),
    ('s_dir','-1','orientation'), ('N_lor','800','elements in Lorentzian region'),
    ('N_pml','200','elements in lower region'), ('N_bg','40','elements in substrate'),
    ('N_x','5','elements across period')]:
    jm.param().set(k, v, d)

comp = jm.component().create('comp1', True)
geom = comp.geom().create('geom1', 2)
r1 = geom.create('r1', 'Rectangle')
r1.set('size', jpype.JArray(jpype.JString)(['Lx', 'L_pml+L_lor+L_bg']))
r1.set('layername', jpype.JArray(jpype.JString)(['low', 'lor']))
r1.setIndex('layer', 'L_pml', 0); r1.setIndex('layer', 'L_lor', 1)
geom.run()

nb_ = int(geom.getNBoundaries())
vc = np.array(geom.getVertexCoord())
adj = [list(np.array(r).ravel()) for r in geom.getAdj(1, 0)]
mid = {b: vc[:, [int(v)-1 for v in adj[b]]].mean(axis=1) for b in range(1, nb_+1)}
ud = np.array(geom.getUpDown())
xs = sorted({round(m[0], 12) for m in mid.values()}); x_lo, x_hi = xs[0], xs[-1]
verts = [b for b, m in mid.items() if abs(m[0]-x_lo) < 1e-12 or abs(m[0]-x_hi) < 1e-12]
horiz = [b for b in mid if b not in verts]
b_top = max(horiz, key=lambda b: mid[b][1]); b_bot = min(horiz, key=lambda b: mid[b][1])
dom_y = {}
for b in verts:
    for d in (int(ud[0][b-1]), int(ud[1][b-1])):
        if d > 0: dom_y.setdefault(d, []).append(mid[b][1])
d_low, d_lor, d_bg = sorted(dom_y, key=lambda d: np.mean(dom_y[d]))
print(f'domains bottom->top: lower/PML={d_low}, Lorentzian={d_lor}, substrate={d_bg}')
print(f'port bnd={b_top}, PEC bnd={b_bot}, verticals={verts}')

m1 = comp.material().create('mat1', 'Common'); m1.label('background')
m1.propertyGroup('def').set('relpermeability', jpype.JArray(jpype.JString)(['1','0','0','0','1','0','0','0','1']))
m1.propertyGroup('def').set('electricconductivity', jpype.JArray(jpype.JString)(['0']*9))
pml = comp.coordSystem().create('pml1', 'PML')
pml.selection().set(jpype.JArray(jpype.JInt)([d_low])); pml.set('PMLfactor', '1')

ph = comp.physics().create('ewfd', 'ElectromagneticWavesFrequencyDomain', 'geom1')
ph.prop('EquationForm').set('form','Frequency'); ph.prop('EquationForm').set('freq_src','userdef')
ph.prop('EquationForm').set('freq','f0'); ph.prop('components').set('components','outofplane')
w1 = ph.feature('wee1')
w1.set('DisplacementFieldModel','DielectricLoss')
w1.set('minput_frequency_src','userdef'); w1.set('minput_frequency','f0')
w1.set('epsilonPrim_mat','userdef'); w1.set('epsilonPrim','n_bg^2')
w1.set('epsilonBis_mat','userdef');  w1.set('epsilonBis','0')

Y0 = 'L_pml + L_lor/2'
DD = f'(((y-({Y0}))^2) + gam^2)'
RE = f'n_bg^2 - s_dir*A_lor*gam*(y-({Y0}))/{DD}'
IM = f'A_lor*gam^2/{DD}'
wl = ph.create('wee_lor','WaveEquationElectric',2); wl.selection().set(jpype.JArray(jpype.JInt)([d_lor]))
wl.label('Lorentzian sKK'); wl.set('DisplacementFieldModel','DielectricLoss')
wl.set('epsilonPrim_mat','userdef'); wl.set('epsilonPrim', RE)
wl.set('epsilonBis_mat','userdef');  wl.set('epsilonBis', IM)
wl.set('mur_mat','userdef'); wl.set('mur','1'); wl.set('sigma_mat','userdef'); wl.set('sigma','0')

pc = ph.create('pc1','PeriodicCondition',1); pc.selection().set(jpype.JArray(jpype.JInt)(verts))
pc.set('PeriodicType','Floquet'); pc.set('Floquet_source','FromPeriodicPort')
pt = ph.create('port1','Port',1); pt.selection().set(jpype.JArray(jpype.JInt)([b_top]))
pt.set('PortType','Periodic'); pt.set('alpha_inc','theta')
pt.set('E0', jpype.JArray(jpype.JInt,2)([[0],[0],[1]]))
pt.set('Eampl', jpype.JArray(jpype.JInt,2)([[0],[0],[1]]))

msh = comp.mesh().create('mesh1','geom1')
left = sorted([b for b in verts if abs(mid[b][0]-x_lo) < 1e-12], key=lambda b: mid[b][1])
for tag, bnd, n in [('e_low',left[0],'N_pml'), ('e_lor',left[1],'N_lor'), ('e_bg',left[2],'N_bg')]:
    e = msh.create(tag,'Edge'); e.selection().set(jpype.JArray(jpype.JInt)([bnd]))
    e.create('dis1','Distribution').set('numelem', n)
e = msh.create('e_x','Edge'); e.selection().set(jpype.JArray(jpype.JInt)([b_bot]))
e.create('dis1','Distribution').set('numelem','N_x')
for i, tag in enumerate(['cp1','cp2','cp3']):
    c = msh.create(tag,'CopyEdge'); c.selection('source').set(jpype.JArray(jpype.JInt)([left[i]]))
    rt = [b for b in verts if abs(b and mid[b][0]-x_hi) < 1e-12 and abs(mid[b][1]-mid[left[i]][1]) < 1e-12]
    c.selection('destination').set(jpype.JArray(jpype.JInt)(rt))
msh.create('map1','Map'); msh.run()

std = jm.study().create('std1'); std.create('stat','Stationary')
std.feature('stat').set('useparam', True)
std.feature('stat').set('pname', jpype.JArray(jpype.JString)(['theta']))
std.feature('stat').set('plistarr', jpype.JArray(jpype.JString)(['range(0,1,89)']))
std.feature('stat').set('punit', jpype.JArray(jpype.JString)(['deg']))
jm.sol().create('sol1').attach('std1'); jm.sol('sol1').createAutoSequence('std1')
model.save(OUTM); print('saved', OUTM, flush=True)

STUDY = model.studies()[0]; out = {}; nelem = {}
def run(key, pml_on, L, A, g_over_lam, s=-1, npml=200):
    model.parameter('L_lor', f'{L}*lam'); model.parameter('A_lor', str(A))
    model.parameter('gam', f'lam/{round(1/g_over_lam)}'); model.parameter('s_dir', str(s))
    n_lor = int(max(400, np.ceil(L/ (g_over_lam/12))))
    model.parameter('N_lor', str(n_lor)); model.parameter('N_pml', str(npml))
    pml.active(bool(pml_on)); model.solve(STUDY)
    R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel(); th = np.linspace(0,89,len(R))
    out[key] = np.vstack([th, R]); nelem[key] = n_lor + npml + 40
    idx = [int(np.argmin(abs(th-t))) for t in [0,20,45,70,89]]
    print(f'{key:26s} N={nelem[key]:>5} R = ' + '  '.join(f'{R[i]:.2e}' for i in idx), flush=True)

print('\n--- orientation check (L=8lam, A=4, gam=lam/5, stacked) ---')
run('orient_s+1', True, 8, 4, 0.2, s=+1)
run('orient_s-1', True, 8, 4, 0.2, s=-1)
S = -1 if out['orient_s-1'][1][0] < out['orient_s+1'][1][0] else +1
print(f'-> reflectionless orientation s = {S:+d}')

print('\n--- Lorentzian ALONE (PEC-backed), A=4, gam=lam/5 ---')
for L in [2,4,8,16]: run(f'alone_L{L}', False, L, 4, 0.2, s=S)
print('\n--- Lorentzian STACKED (+PML), A=4, gam=lam/5 ---')
for L in [2,4,8,16]: run(f'stack_L{L}', True, L, 4, 0.2, s=S)
print('\n--- stacked, A sweep at L=8lam (gam=lam/5) ---')
for A in [1,2,8]: run(f'stack_A{A}', True, 8, A, 0.2, s=S)
print('\n--- constant A*gam = 0.8*lam, trade A vs gam, L=8lam, stacked ---')
for A, g in [(8,0.1),(4,0.2),(2,0.4),(1,0.8)]: run(f'const_Ag_A{A}', True, 8, A, g, s=S)

print('\n--- PML alone, generous mesh ---')
model.parameter('A_lor','0')
for pf in [1,4]:
    pml.set('PMLfactor', str(pf)); model.parameter('L_lor','2*lam')
    model.parameter('N_lor','400'); model.parameter('N_pml','800')
    pml.active(True); model.solve(STUDY)
    R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel(); th = np.linspace(0,89,len(R))
    out[f'PML_pf{pf}'] = np.vstack([th,R]); nelem[f'PML_pf{pf}'] = 800+400+40
    idx=[int(np.argmin(abs(th-t))) for t in [0,20,45,70,89]]
    print(f'PML pf={pf} (N_pml=800)      N={nelem[f"PML_pf{pf}"]:>5} R = '+'  '.join(f'{R[i]:.2e}' for i in idx), flush=True)

np.savez(os.path.join(HERE,'lorentzian_stacked.npz'),
         nelem=np.array([[k, nelem[k]] for k in nelem], dtype=object), **out)
model.save(OUTM); client.clear(); print('\ndone')
