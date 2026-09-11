"""Fig 9 redone consistently: ALL profiles keep a FIXED absolute feature width (lam/5),
and all use the same element scaling. Only the domain length varies -> pure truncation."""
import os, sys, time, numpy as np, jpype, mph
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,ROOT); import tmm_helper as tmm_h
MODEL=os.path.join(HERE, 'skk_master.mph'); OUT=os.path.join(HERE,'master_data.npz')
lam=0.8e-6; NB=1.7; EB=NB**2; NPTS=6001; A80='80'; WFIX=lam/5

def profile(kind,L):
    y=np.linspace(0.0,L,NPTS)
    if kind=='logistic':
        er=1.0+(EB-1.0)/(1.0+np.exp(-(y-L/2)/WFIX)); return y,er,-tmm_h.ht_derivative(y,er),1.0
    u=y-L/2; D=u**2+WFIX**2; A=4.0
    if kind=='lorentzian': return y, EB+A*WFIX*u/D, A*WFIX**2/D, EB
    if kind=='tapered':    return y, EB+A*WFIX*u/D/np.cosh(u/(0.5*lam)), A*WFIX**2/D, EB

t0=time.time(); client=mph.start(); model=client.load(MODEL); jm=model.java
comp=jm.component('comp1'); pml=comp.coordSystem('pml1'); STUDY=model.studies()[0]
old=dict(np.load(OUT)); RES={}
LG=sorted(set(round(float(v),3) for v in np.geomspace(1,100,21)))
print(f'{"L/lam":>8} {"N":>6}  ' + '  '.join(f'{k:>11}' for k in ['logistic','lorentzian','tapered']),flush=True)
for L in LG:
    N=int(min(8000,max(400,50*L))); row=[]
    jm.study('std1').feature('stat').set('plistarr',jpype.JArray(jpype.JString)([A80]))
    model.parameter('L_term',f'{L}*lam'); model.parameter('L_low','2*lam')
    model.parameter('N_term',str(N)); model.parameter('N_low','400')
    for k in ['logistic','lorentzian','tapered']:
        y,er,ei,el=profile(k,L*lam)
        comp.func('i_re').set('table',jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,er)]))
        comp.func('i_im').set('table',jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,ei)]))
        model.parameter('eps_low',repr(float(el))); pml.active(False)
        model.solve(STUDY)
        r=float(np.asarray(model.evaluate('ewfd.Rtotal')).ravel()[0])
        RES[f'P4_TH_{k}_L{L}']=np.array([[80.0],[r]]); row.append(r)
    print(f'{L:8.2f} {N:6d}  ' + '  '.join(f'{v:11.3e}' for v in row),flush=True)
old.update(RES); np.savez(OUT,**old); client.clear(); print(f'done {time.time()-t0:.0f}s')
