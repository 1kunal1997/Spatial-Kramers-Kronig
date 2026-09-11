"""Unified terminator study. ONE model, one mesh, one port; only the profile changes.

Geometry (bottom -> top):  [lower region / PML] | [terminator] | [substrate] , PEC at bottom,
periodic port on top, Floquet sides.  eps of the terminator comes from interpolation tables
so any profile can be dropped in and every comparison is genuinely apples-to-apples.
"""
import os, sys, time, numpy as np, jpype, mph
from scipy.signal import hilbert
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); import tmm_helper as tmm_h
OUT  = os.path.join(HERE, 'master_data.npz')
OUTM = os.path.join(HERE, 'skk_master.mph')

lam = 0.8e-6; NB = 1.7; EB = NB**2
FINE = 'range(0,1,89)'
COARSE = '0 10 20 30 40 50 60 70 80 89'
NPTS = 3001

# ---------------------------------------------------------------- profiles
def profile(kind, L, A=4.0, gam=None, w=None, ell=None):
    """local y in [0, L]; y=L is the ENTRY face (substrate side), y=0 is the PEC side."""
    gam = gam if gam is not None else lam/5
    w   = w   if w   is not None else L/40
    ell = ell if ell is not None else 0.5*lam
    y = np.linspace(0.0, L, NPTS)
    if kind == 'logistic':
        er = 1.0 + (EB-1.0)/(1.0+np.exp(-(y-L/2)/w))      # EB at entry -> 1 at PEC
        ei = -tmm_h.ht_derivative(y, er)
        return y, er, ei, 1.0
    u = y - L/2; D = u**2 + gam**2
    if kind == 'lorentzian':
        return y, EB + A*gam*u/D, A*gam**2/D, EB
    if kind == 'tapered':
        return y, EB + A*gam*u/D/np.cosh(u/ell), A*gam**2/D, EB
    if kind == 'gain':
        W = 400*lam; NW = 2**20
        uu = np.linspace(-W/2, W/2, NW); DD = uu**2+gam**2
        dre = A*gam*uu/DD/np.cosh(uu/ell)
        im  = np.imag(hilbert(dre))
        return y, EB + np.interp(u, uu, dre), np.interp(u, uu, im), EB
    raise ValueError(kind)

PRETTY = {'logistic':'logistic (DiHiTI)', 'lorentzian':'Lorentzian',
          'tapered':'tapered Lorentzian', 'gain':'tapered + exact HT (gain)'}

# ---------------------------------------------------------------- model
t0 = time.time()
client = mph.start(); model = client.create('skk_master'); jm = model.java
for k, v, d in [('lam','800[nm]',''), ('f0','c_const/lam',''), ('theta','0[deg]',''),
    ('n_bg','1.7',''), ('Lx','lam/4',''), ('L_bg','lam',''), ('L_term','8*lam',''),
    ('L_low','lam',''), ('eps_low','n_bg^2',''), ('N_term','800',''), ('N_low','200',''),
    ('N_bg','40',''), ('N_x','5','')]:
    jm.param().set(k, v, d)
comp = jm.component().create('comp1', True)
geom = comp.geom().create('geom1', 2)
r1 = geom.create('r1','Rectangle')
r1.set('size', jpype.JArray(jpype.JString)(['Lx','L_low+L_term+L_bg']))
r1.set('layername', jpype.JArray(jpype.JString)(['low','term']))
r1.setIndex('layer','L_low',0); r1.setIndex('layer','L_term',1)
geom.run()

nb_ = int(geom.getNBoundaries()); vc = np.array(geom.getVertexCoord())
adj = [list(np.array(r).ravel()) for r in geom.getAdj(1,0)]
mid = {b: vc[:,[int(v)-1 for v in adj[b]]].mean(axis=1) for b in range(1,nb_+1)}
ud = np.array(geom.getUpDown())
xs = sorted({round(m[0],12) for m in mid.values()}); xlo,xhi = xs[0],xs[-1]
verts = [b for b,m in mid.items() if abs(m[0]-xlo)<1e-12 or abs(m[0]-xhi)<1e-12]
horiz = [b for b in mid if b not in verts]
b_top = max(horiz,key=lambda b:mid[b][1]); b_bot = min(horiz,key=lambda b:mid[b][1])
dy={}
for b in verts:
    for d in (int(ud[0][b-1]),int(ud[1][b-1])):
        if d>0: dy.setdefault(d,[]).append(mid[b][1])
d_low,d_term,d_bg = sorted(dy,key=lambda d:np.mean(dy[d]))
print(f'domains low={d_low} term={d_term} bg={d_bg}; port={b_top} pec={b_bot}', flush=True)

m1 = comp.material().create('mat1','Common'); m1.label('background')
m1.propertyGroup('def').set('relpermeability', jpype.JArray(jpype.JString)(['1','0','0','0','1','0','0','0','1']))
m1.propertyGroup('def').set('electricconductivity', jpype.JArray(jpype.JString)(['0']*9))
pml = comp.coordSystem().create('pml1','PML')
pml.selection().set(jpype.JArray(jpype.JInt)([d_low])); pml.set('PMLfactor','1')

ph = comp.physics().create('ewfd','ElectromagneticWavesFrequencyDomain','geom1')
ph.prop('EquationForm').set('form','Frequency'); ph.prop('EquationForm').set('freq_src','userdef')
ph.prop('EquationForm').set('freq','f0'); ph.prop('components').set('components','outofplane')
w1 = ph.feature('wee1'); w1.set('DisplacementFieldModel','DielectricLoss')
w1.set('minput_frequency_src','userdef'); w1.set('minput_frequency','f0')
w1.set('epsilonPrim_mat','userdef'); w1.set('epsilonPrim','n_bg^2')
w1.set('epsilonBis_mat','userdef');  w1.set('epsilonBis','0')

def mkfun(tag, fn):
    f = comp.func().create(tag,'Interpolation'); f.set('source','table')
    f.set('table', jpype.JArray(jpype.JString,2)([['0','0'],['1','0']]))
    f.set('funcname',fn); f.set('interp','linear'); f.set('extrap','const')
    f.set('argunit','m'); f.set('fununit','1'); return f
f_re = mkfun('i_re','p_re'); f_im = mkfun('i_im','p_im')

wlow = ph.create('wee_low','WaveEquationElectric',2); wlow.selection().set(jpype.JArray(jpype.JInt)([d_low]))
wlow.label('lower region'); wlow.set('DisplacementFieldModel','DielectricLoss')
wlow.set('epsilonPrim_mat','userdef'); wlow.set('epsilonPrim','eps_low')
wlow.set('epsilonBis_mat','userdef');  wlow.set('epsilonBis','0')
wlow.set('mur_mat','userdef'); wlow.set('mur','1'); wlow.set('sigma_mat','userdef'); wlow.set('sigma','0')

wt = ph.create('wee_term','WaveEquationElectric',2); wt.selection().set(jpype.JArray(jpype.JInt)([d_term]))
wt.label('terminator'); wt.set('DisplacementFieldModel','DielectricLoss')
wt.set('epsilonPrim_mat','userdef'); wt.set('epsilonPrim','p_re(y-L_low)')
wt.set('epsilonBis_mat','userdef');  wt.set('epsilonBis','p_im(y-L_low)')
wt.set('mur_mat','userdef'); wt.set('mur','1'); wt.set('sigma_mat','userdef'); wt.set('sigma','0')

pc = ph.create('pc1','PeriodicCondition',1); pc.selection().set(jpype.JArray(jpype.JInt)(verts))
pc.set('PeriodicType','Floquet'); pc.set('Floquet_source','FromPeriodicPort')
pt = ph.create('port1','Port',1); pt.selection().set(jpype.JArray(jpype.JInt)([b_top]))
pt.set('PortType','Periodic'); pt.set('alpha_inc','theta')
pt.set('E0', jpype.JArray(jpype.JInt,2)([[0],[0],[1]]))
pt.set('Eampl', jpype.JArray(jpype.JInt,2)([[0],[0],[1]]))

msh = comp.mesh().create('mesh1','geom1')
left = sorted([b for b in verts if abs(mid[b][0]-xlo)<1e-12], key=lambda b: mid[b][1])
for tag,bnd,n in [('e_low',left[0],'N_low'),('e_term',left[1],'N_term'),('e_bg',left[2],'N_bg')]:
    e = msh.create(tag,'Edge'); e.selection().set(jpype.JArray(jpype.JInt)([bnd]))
    e.create('dis1','Distribution').set('numelem', n)
e = msh.create('e_x','Edge'); e.selection().set(jpype.JArray(jpype.JInt)([b_bot]))
e.create('dis1','Distribution').set('numelem','N_x')
for i,tag in enumerate(['cp1','cp2','cp3']):
    c = msh.create(tag,'CopyEdge'); c.selection('source').set(jpype.JArray(jpype.JInt)([left[i]]))
    rt=[b for b in verts if abs(mid[b][0]-xhi)<1e-12 and abs(mid[b][1]-mid[left[i]][1])<1e-12]
    c.selection('destination').set(jpype.JArray(jpype.JInt)(rt))
msh.create('map1','Map')

std = jm.study().create('std1'); std.create('stat','Stationary')
std.feature('stat').set('useparam',True)
std.feature('stat').set('pname', jpype.JArray(jpype.JString)(['theta']))
std.feature('stat').set('plistarr', jpype.JArray(jpype.JString)([FINE]))
std.feature('stat').set('punit', jpype.JArray(jpype.JString)(['deg']))
jm.sol().create('sol1').attach('std1'); jm.sol('sol1').createAutoSequence('std1')
model.save(OUTM); print('model built', flush=True)

STUDY = model.studies()[0]; RES = {}; n_done = [0]
def push(y, er, ei):
    comp.func('i_re').set('table', jpype.JArray(jpype.JString,2)(
        [[repr(float(a)),repr(float(b))] for a,b in zip(y,er)]))
    comp.func('i_im').set('table', jpype.JArray(jpype.JString,2)(
        [[repr(float(a)),repr(float(b))] for a,b in zip(y,ei)]))

def solve(key, kind=None, L=8*lam, pml_on=False, pf=1, N_term=800, N_low=200,
          fine=False, L_low=lam, **kw):
    jm.study('std1').feature('stat').set('plistarr',
        jpype.JArray(jpype.JString)([FINE if fine else COARSE]))
    model.parameter('L_term', f'{L/lam}*lam'); model.parameter('L_low', f'{L_low/lam}*lam')
    model.parameter('N_term', str(int(N_term))); model.parameter('N_low', str(int(N_low)))
    if kind is None:
        push(np.array([0.0, L]), np.array([EB,EB]), np.array([0.0,0.0]))
        model.parameter('eps_low','n_bg^2')
    else:
        y, er, ei, elow = profile(kind, L, **kw)
        push(y, er, ei); model.parameter('eps_low', repr(float(elow)))
    pml.active(bool(pml_on)); pml.set('PMLfactor', str(pf))
    try:
        model.solve(STUDY)
        R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
    except Exception as e:
        print(f'  !! {key}: {str(e).splitlines()[0][:60]}', flush=True); return
    th = np.array([float(v) for v in (np.linspace(0,89,90) if fine
                   else np.array([0,10,20,30,40,50,60,70,80,89]))])
    RES[key] = np.vstack([th[:len(R)], R])
    RES[key+'__meta'] = np.array([N_term+N_low, L/lam, L_low/lam, 1 if pml_on else 0, pf], float)
    n_done[0] += 1
    print(f'  [{n_done[0]:3d}] {key:38s} N={int(N_term+N_low):5d} R(0)={R[0]:.3e} R(89)={R[-1]:.3f}'
          f'  [{time.time()-t0:.0f}s]', flush=True)

KINDS = ['logistic','lorentzian','tapered','gain']

print('\n=== S1 PML: angle curves (fine) ===', flush=True)
for pf in [1,2,4]: solve(f'PML_pf{pf}', None, L=2*lam, pml_on=True, pf=pf,
                         N_term=40, N_low=800, fine=True, L_low=2*lam)
print('\n=== S2 PML: element sweep (A-E "D") ===', flush=True)
for pf in [1,2,4]:
    for N in [20,50,100,200,400,800,1600]:
        solve(f'PMLN_pf{pf}_N{N}', None, L=2*lam, pml_on=True, pf=pf,
              N_term=20, N_low=N, L_low=2*lam)
print('\n=== S3 PML: thickness sweep (A-E "B") ===', flush=True)
for Lm in [1,2,4,8,16]:
    solve(f'PMLL_L{Lm}', None, L=2*lam, pml_on=True, pf=1, N_term=20, N_low=400, L_low=Lm*lam)
print('\n=== S4 profiles at L=8lam (fine, headline) ===', flush=True)
for k in KINDS: solve(f'FINE_{k}', k, L=8*lam, pml_on=False, N_term=960, N_low=200, fine=True)
print('\n=== S5 fixed-thickness comparisons (A-E "A") ===', flush=True)
for Lm in [2,4,6,12,16]:
    for k in KINDS:
        solve(f'L{Lm}_{k}', k, L=Lm*lam, pml_on=False,
              N_term=int(max(400,120*Lm)), N_low=200)
print('\n=== S6 backing isolation (A-E "E") ===', flush=True)
for k in KINDS:
    for Lm in [2,8]:
        solve(f'BACK_{k}_L{Lm}_stack', k, L=Lm*lam, pml_on=True, pf=1,
              N_term=int(max(400,120*Lm)), N_low=400)
print('\n=== S7 element sweep, sKK (A-E "D") ===', flush=True)
for k in ['logistic','tapered']:
    for N in [50,100,200,400,800,1600]:
        solve(f'SKKN_{k}_N{N}', k, L=8*lam, pml_on=False, N_term=N, N_low=20)
print('\n=== S8 stacking ratio at fixed total 8lam (A-E "C") ===', flush=True)
for k in ['logistic','lorentzian']:
    for r in [0.1,0.25,0.5,0.75,0.9]:
        solve(f'RATIO_{k}_r{r}', k, L=r*8*lam, pml_on=True, pf=1,
              N_term=800, N_low=200, L_low=(1-r)*8*lam)
print('\n=== S9 parameter sweeps at fixed L=8lam ===', flush=True)
for A in [1,2,4,8,16]:
    solve(f'PAR_lorA{A}', 'lorentzian', L=8*lam, N_term=960, A=A)
    solve(f'PAR_tapA{A}', 'tapered',    L=8*lam, N_term=960, A=A)
for g in [10,5,2.5,1.25]:
    solve(f'PAR_lorG{g}', 'lorentzian', L=8*lam, N_term=960, gam=lam/g)
for wd in [80,40,20,10]:
    solve(f'PAR_logW{wd}', 'logistic', L=8*lam, N_term=960, w=8*lam/wd)
for el in [0.25,0.5,1,2,4]:
    solve(f'PAR_tapE{el}', 'tapered', L=8*lam, N_term=960, ell=el*lam)

prof = {}
for k in KINDS:
    y,er,ei,_ = profile(k, 8*lam)
    prof[f'PROF_{k}'] = np.vstack([y/lam, er, ei])
np.savez(OUT, **RES, **prof)
model.save(OUTM); client.clear()
print(f'\nDONE {n_done[0]} solves in {time.time()-t0:.0f}s -> {OUT}', flush=True)
