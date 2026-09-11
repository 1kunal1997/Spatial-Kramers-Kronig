"""Figs 13/14: rerun the logistic with a FIXED w = lam/5 so its feature width matches
the Lorentzian family's gamma at every thickness (at L=8lam the old L/40 already equalled lam/5)."""
import os, sys, numpy as np, jpype, mph
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,ROOT); import tmm_helper as tmm_h
MODEL=os.path.join(HERE, 'skk_master.mph'); OUT=os.path.join(HERE,'master_data.npz')
lam=0.8e-6; EB=1.7**2; WFIX=lam/5; FINE='range(0,1,89)'
client=mph.start(); model=client.load(MODEL); jm=model.java
comp=jm.component('comp1'); pml=comp.coordSystem('pml1'); STUDY=model.studies()[0]
old=dict(np.load(OUT)); RES={}
for Lm in [2,12]:
    L=Lm*lam; y=np.linspace(0.0,L,6001)
    er=1.0+(EB-1.0)/(1.0+np.exp(-(y-L/2)/WFIX)); ei=-tmm_h.ht_derivative(y,er)
    jm.study('std1').feature('stat').set('plistarr',jpype.JArray(jpype.JString)([FINE]))
    model.parameter('L_term',f'{Lm}*lam'); model.parameter('L_low','2*lam')
    model.parameter('N_term','800'); model.parameter('N_low','800'); model.parameter('eps_low','1')
    comp.func('i_re').set('table',jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,er)]))
    comp.func('i_im').set('table',jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,ei)]))
    pml.active(False); model.solve(STUDY)
    R=np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
    RES[f'P2_M{Lm}_logistic']=np.vstack([np.linspace(0,89,90)[:len(R)],R])
    o=old[f'P2_M{Lm}_logistic'][1]
    print(f'L={Lm}lam  w=lam/5: R(0)={R[0]:.3e} R(80)={R[80]:.3e}   (was w=L/40: {o[0]:.3e}, {o[80]:.3e})',flush=True)
old.update(RES); np.savez(OUT,**old); client.clear(); print('done')
