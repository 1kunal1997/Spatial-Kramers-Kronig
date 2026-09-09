"""Exact-HT tapered Lorentzian, gain allowed. eps'' = HT[eps'_tapered] (integrates to 0)."""
import os, sys, numpy as np, jpype, mph
from scipy.signal import hilbert
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,ROOT); import tmm_helper as tmm_h
MODEL=os.path.join(ROOT,'Comsol Files','skk_lorentzian_stacked.mph')
OUTM =os.path.join(ROOT,'Comsol Files','skk_lorentzian_gain.mph')

lam=0.8e-6; eb=1.7**2; A=4.0; gam=lam/5; L=8*lam
W=400*lam; N=2**21
u=np.linspace(-W/2,W/2,N); D=u**2+gam**2

def tables(ell, mode):
    dre = -A*gam*u/D/np.cosh(u/ell)
    if mode=='HT':      im = np.imag(hilbert(dre))
    elif mode=='LOR':   im = A*gam**2/D
    m=(u>=-L/2)&(u<=L/2)
    if mode=='DIHITI':  im=np.zeros_like(u); im[m]=tmm_h.ht_derivative(u[m], eb+dre[m])
    return u[m]/lam, dre[m], im[m]

client=mph.start(); model=client.load(MODEL); jm=model.java; comp=jm.component('comp1')
jm.param().set('ell','0.5*lam','taper length')
xi,dre0,im0=tables(0.5*lam,'HT')
def mk(tag,fn,vs):
    f=comp.func().create(tag,'Interpolation'); f.set('source','table')
    f.set('table', jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(xi,vs)]))
    f.set('funcname',fn); f.set('interp','linear'); f.set('extrap','const')
    f.set('argunit','1'); f.set('fununit','1')
mk('t_re','g_re',dre0); mk('t_im','g_im',im0)
Y0='L_pml + L_lor/2'
wl=comp.physics('ewfd').feature('wee_lor')
wl.set('epsilonPrim', f'n_bg^2 + s_dir*g_re((y-({Y0}))/lam)')
wl.set('epsilonBis',  f'g_im((y-({Y0}))/lam)')  # eps'' is EVEN in u -> no s_dir
model.parameter('L_lor','8*lam'); model.parameter('A_lor','4'); model.parameter('gam','lam/5')
model.parameter('s_dir','-1'); model.parameter('N_lor','960'); model.parameter('N_pml','400')

pml=comp.coordSystem('pml1'); STUDY=model.studies()[0]; out={}
def setab(ell,mode):
    _,a,b=tables(ell*lam,mode)
    comp.func('t_re').set('table', jpype.JArray(jpype.JString,2)([[repr(float(p)),repr(float(q))] for p,q in zip(xi,a)]))
    comp.func('t_im').set('table', jpype.JArray(jpype.JString,2)([[repr(float(p)),repr(float(q))] for p,q in zip(xi,b)]))
    return b
def run(key,ell,mode,pml_on):
    b=setab(ell,mode); pml.active(bool(pml_on))
    try: model.solve(STUDY)
    except Exception as e: print(f'{key:24s} SOLVE FAILED: {str(e).splitlines()[0][:70]}'); return
    R=np.asarray(model.evaluate('ewfd.Rtotal')).ravel(); th=np.linspace(0,89,len(R))
    Ab=np.asarray(model.evaluate('ewfd.Atotal')).ravel()
    out[key]=np.vstack([th,R])
    idx=[int(np.argmin(abs(th-t))) for t in [0,20,45,70,89]]
    flag=' <-- R>1 !' if R.max()>1.0 else ''
    print(f'{key:24s} min_eps_im={b.min():+.3f} R = '+'  '.join(f'{R[i]:.3e}' for i in idx)
          +f'  maxR={R.max():.3e}{flag}', flush=True)

print('--- exact-HT profile (gain allowed), PEC-backed ---')
for e in [0.5,1.0,2.0]: run(f'HT_ell{e}_alone', e, 'HT', False)
print('\n--- exact-HT profile, stacked (+PML) ---')
for e in [0.5,1.0,2.0]: run(f'HT_ell{e}_stack', e, 'HT', True)
print('\n--- DiHiTI partner (partial gain), PEC-backed ---')
for e in [0.5,2.0]: run(f'DIH_ell{e}_alone', e, 'DIHITI', False)
print('\n--- control: same eps_re, passive Lorentzian eps_im ---')
run('LOR_ell0.5_alone', 0.5, 'LOR', False)

np.savez(os.path.join(HERE,'lorentzian_gain.npz'), **out)
model.save(OUTM); client.clear(); print('\nsaved', OUTM)
