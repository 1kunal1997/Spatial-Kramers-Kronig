"""Data patch: matched-mesh comparisons, extended sweeps at theta=80, corrected gain profile."""
import os, sys, time, numpy as np, jpype, mph
from scipy.signal import hilbert
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,ROOT); import tmm_helper as tmm_h
MODEL=os.path.join(ROOT,'Comsol Files','skk_master.mph'); OUT=os.path.join(HERE,'master_data.npz')
lam=0.8e-6; NB=1.7; EB=NB**2; NPTS=4001
FINE='range(0,1,89)'; COARSE='0 10 20 30 40 50 60 70 80 89'; A80='80'

def profile(kind,L,A=4.0,gam=None,w=None,ell=None):
    gam=gam or lam/5; w=w or L/40; ell=ell or 0.5*lam
    y=np.linspace(0.0,L,NPTS)
    if kind=='logistic':
        er=1.0+(EB-1.0)/(1.0+np.exp(-(y-L/2)/w)); return y,er,-tmm_h.ht_derivative(y,er),1.0
    u=y-L/2; D=u**2+gam**2
    if kind=='lorentzian': return y, EB+A*gam*u/D, A*gam**2/D, EB
    if kind=='tapered':    return y, EB+A*gam*u/D/np.cosh(u/ell), A*gam**2/D, EB
    if kind=='gain':
        W=400*lam; NW=2**20; uu=np.linspace(-W/2,W/2,NW); DD=uu**2+gam**2
        dre=A*gam*uu/DD/np.cosh(uu/ell)
        im=-np.imag(hilbert(dre))          # FIX: match the passive convention eps''=-HT[eps'_pert]
        return y, EB+np.interp(u,uu,dre), np.interp(u,uu,im), EB
    raise ValueError(kind)

t0=time.time(); client=mph.start(); model=client.load(MODEL); jm=model.java
comp=jm.component('comp1'); pml=comp.coordSystem('pml1'); STUDY=model.studies()[0]
old=dict(np.load(OUT)); RES={}; n=[0]

def push(y,er,ei):
    comp.func('i_re').set('table', jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,er)]))
    comp.func('i_im').set('table', jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,ei)]))

def solve(key,kind,L,pml_on,pf,N_term,N_low,ang,L_low=2*lam,**kw):
    jm.study('std1').feature('stat').set('plistarr',jpype.JArray(jpype.JString)([ang]))
    model.parameter('L_term',f'{L/lam}*lam'); model.parameter('L_low',f'{L_low/lam}*lam')
    model.parameter('N_term',str(int(N_term))); model.parameter('N_low',str(int(N_low)))
    if kind is None:
        push(np.array([0.0,L]),np.array([EB,EB]),np.array([0.0,0.0])); model.parameter('eps_low','n_bg^2')
    else:
        y,er,ei,el=profile(kind,L,**kw); push(y,er,ei); model.parameter('eps_low',repr(float(el)))
    pml.active(bool(pml_on)); pml.set('PMLfactor',str(pf))
    try: model.solve(STUDY); R=np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
    except Exception as e: print(f'  !! {key}: {str(e).splitlines()[0][:60]}',flush=True); return
    th = (np.linspace(0,89,90) if ang==FINE else
          np.array([0,10,20,30,40,50,60,70,80,89],float) if ang==COARSE else np.array([80.0]))
    RES[key]=np.vstack([th[:len(R)],R]); n[0]+=1
    if n[0]%15==0 or ang==FINE:
        print(f'  [{n[0]:3d}] {key:34s} R={R[0]:.3e}  [{time.time()-t0:.0f}s]',flush=True)

KIN=['logistic','lorentzian','tapered']
NEL=sorted(set(int(round(v)) for v in np.geomspace(5,1000,22)))
print(f'element grid: {NEL}',flush=True)

print('\n=== P1 PML element sweep 5..1000 (figs 3, 3b) ===',flush=True)
for pf in [1,2,4]:
    for N in NEL: solve(f'P2_PMLN_pf{pf}_N{N}',None,2*lam,True,pf,400,N,COARSE)

print('\n=== P2 matched-mesh comparisons (figs 7, 13, 14) ===',flush=True)
for Lm in [2,8,12]:
    for k in KIN:  solve(f'P2_M{Lm}_{k}',k,Lm*lam,False,1,800,800,FINE)
    for pf in [1,4]: solve(f'P2_M{Lm}_PML{pf}',None,Lm*lam,True,pf,800,800,FINE)

print('\n=== P3 R vs thickness at theta=80 (fig 9) ===',flush=True)
LGRID=sorted(set(round(float(v),3) for v in np.geomspace(1,100,21)))
for k in KIN:
    for Lm in LGRID:
        Nt = int(min(6000,max(400, 50*Lm if k!='logistic' else 400)))
        solve(f'P2_TH_{k}_L{Lm}',k,Lm*lam,False,1,Nt,400,A80)
solve('P2_TH_PML',None,2*lam,True,1,400,400,A80)

print('\n=== P4 R vs elements at theta=80 (fig 10) ===',flush=True)
for k in KIN:
    for N in NEL: solve(f'P2_EL_{k}_N{N}',k,8*lam,False,1,N,400,A80)

print('\n=== P5 corrected gain profile (fig 18) ===',flush=True)
solve('P2_gain_fine','gain',8*lam,False,1,960,400,FINE)
y,er,ei,_=profile('gain',8*lam)
RES['P2_PROF_gain']=np.vstack([y/lam,er,ei])
print(f'  corrected gain eps_im: min={ei.min():+.3f} max={ei.max():+.3f}',flush=True)

old.update(RES); np.savez(OUT,**old); client.clear()
print(f'\nDONE {n[0]} solves in {time.time()-t0:.0f}s',flush=True)
