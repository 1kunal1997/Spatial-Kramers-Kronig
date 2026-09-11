"""Re-verify convergence for fig 9 with the CURRENT fixed-width (lam/5) profiles."""
import os, sys, numpy as np, jpype, mph
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,ROOT); import tmm_helper as tmm_h
MODEL=os.path.join(HERE, 'skk_master.mph')
lam=0.8e-6; EB=1.7**2; WFIX=lam/5; A80='80'
def profile(kind,L):
    y=np.linspace(0.0,L,6001)
    if kind=='logistic':
        er=1.0+(EB-1.0)/(1.0+np.exp(-(y-L/2)/WFIX)); return y,er,-tmm_h.ht_derivative(y,er),1.0
    u=y-L/2; D=u**2+WFIX**2; A=4.0
    if kind=='lorentzian': return y, EB+A*WFIX*u/D, A*WFIX**2/D, EB
    if kind=='tapered':    return y, EB+A*WFIX*u/D/np.cosh(u/(0.5*lam)), A*WFIX**2/D, EB
client=mph.start(); model=client.load(MODEL); jm=model.java
comp=jm.component('comp1'); pml=comp.coordSystem('pml1'); STUDY=model.studies()[0]
def run(k,L,N):
    jm.study('std1').feature('stat').set('plistarr',jpype.JArray(jpype.JString)([A80]))
    model.parameter('L_term',f'{L}*lam'); model.parameter('L_low','2*lam')
    model.parameter('N_term',str(int(N))); model.parameter('N_low','400')
    y,er,ei,el=profile(k,L*lam)
    comp.func('i_re').set('table',jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,er)]))
    comp.func('i_im').set('table',jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,ei)]))
    model.parameter('eps_low',repr(float(el))); pml.active(False); model.solve(STUDY)
    return float(np.asarray(model.evaluate('ewfd.Rtotal')).ravel()[0])
print(f'{"profile":11s} {"L/lam":>7} {"N used":>7} {"R":>11} | {"2N":>6} {"R":>11} | {"4N":>6} {"R":>11} | {"max dev":>8}')
worst=0
for k in ['logistic','lorentzian','tapered']:
    for L in [1.0,2.0,8.0,31.62,100.0]:
        N=int(min(8000,max(400,50*L)))
        r=[run(k,L,N),run(k,L,2*N),run(k,L,4*N)]
        dev=max(abs(x/r[2]-1) for x in r[:2])*100; worst=max(worst,dev)
        print(f'{k:11s} {L:7.2f} {N:7d} {r[0]:11.3e} | {2*N:6d} {r[1]:11.3e} | {4*N:6d} {r[2]:11.3e} | {dev:7.2f}%',flush=True)
print(f'\nworst deviation from the 4N reference across all points: {worst:.2f}%')
client.clear()
