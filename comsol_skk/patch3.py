"""Convergence check for fig 9, plus PML pf=4 reference at theta=80."""
import os, sys, time, numpy as np, jpype, mph
from scipy.signal import hilbert
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,ROOT); import tmm_helper as tmm_h
MODEL=os.path.join(ROOT,'Comsol Files','skk_master.mph'); OUT=os.path.join(HERE,'master_data.npz')
lam=0.8e-6; NB=1.7; EB=NB**2; NPTS=4001; A80='80'
def profile(kind,L,A=4.0,gam=None,w=None,ell=None):
    gam=gam or lam/5; w=w or L/40; ell=ell or 0.5*lam
    y=np.linspace(0.0,L,NPTS)
    if kind=='logistic':
        er=1.0+(EB-1.0)/(1.0+np.exp(-(y-L/2)/w)); return y,er,-tmm_h.ht_derivative(y,er),1.0
    u=y-L/2; D=u**2+gam**2
    if kind=='lorentzian': return y, EB+A*gam*u/D, A*gam**2/D, EB
    if kind=='tapered':    return y, EB+A*gam*u/D/np.cosh(u/ell), A*gam**2/D, EB
t0=time.time(); client=mph.start(); model=client.load(MODEL); jm=model.java
comp=jm.component('comp1'); pml=comp.coordSystem('pml1'); STUDY=model.studies()[0]
old=dict(np.load(OUT)); RES={}
def run(kind,L,N,pml_on=False,pf=1):
    jm.study('std1').feature('stat').set('plistarr',jpype.JArray(jpype.JString)([A80]))
    model.parameter('L_term',f'{L}*lam'); model.parameter('L_low','2*lam')
    model.parameter('N_term',str(int(N))); model.parameter('N_low','400')
    if kind is None:
        comp.func('i_re').set('table',jpype.JArray(jpype.JString,2)([['0',repr(EB)],[repr(L*lam),repr(EB)]]))
        comp.func('i_im').set('table',jpype.JArray(jpype.JString,2)([['0','0'],[repr(L*lam),'0']]))
        model.parameter('eps_low','n_bg^2')
    else:
        y,er,ei,el=profile(kind,L*lam)
        comp.func('i_re').set('table',jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,er)]))
        comp.func('i_im').set('table',jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,ei)]))
        model.parameter('eps_low',repr(float(el)))
    pml.active(bool(pml_on)); pml.set('PMLfactor',str(pf))
    model.solve(STUDY); return float(np.asarray(model.evaluate('ewfd.Rtotal')).ravel()[0])

print(f'{"profile":12s} {"L/lam":>7} {"N used":>7} {"R":>11} {"N x4":>7} {"R":>11} {"ratio":>7}')
conv={}
for k in ['logistic','lorentzian','tapered']:
    for L in [10.0,31.62,100.0]:
        Nb = 400 if k=='logistic' else int(min(6000,max(400,50*L)))
        r1=run(k,L,Nb); r2=run(k,L,min(12000,Nb*4))
        conv[f'{k}_L{L}']=np.array([Nb,r1,Nb*4,r2])
        print(f'{k:12s} {L:7.2f} {Nb:7d} {r1:11.3e} {Nb*4:7d} {r2:11.3e} {r1/r2:7.2f}',flush=True)
for pf in [1,4]:
    RES[f'P3_PMLref_pf{pf}']=np.array([[80.0],[run(None,2.0,400,True,pf)]])
    print(f'PML pf={pf} at theta=80: {RES[f"P3_PMLref_pf{pf}"][1][0]:.3e}',flush=True)
RES.update({f'P3_CONV_{k}':v for k,v in conv.items()})
old.update(RES); np.savez(OUT,**old); client.clear(); print(f'done {time.time()-t0:.0f}s')
