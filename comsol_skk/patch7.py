"""Equal-cost element sweeps at theta = 0, 45, 80, 89 for all terminators + PML."""
import os, sys, time, numpy as np, jpype, mph
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,ROOT); import tmm_helper as tmm_h
MODEL=os.path.join(ROOT,'Comsol Files','skk_master.mph'); OUT=os.path.join(HERE,'master_data.npz')
lam=0.8e-6; EB=1.7**2; WFIX=lam/5; ANG='0 45 80 89'; THS=[0.,45.,80.,89.]
L=8*lam
def profile(kind):
    y=np.linspace(0.0,L,6001)
    if kind=='logistic':
        er=1.0+(EB-1.0)/(1.0+np.exp(-(y-L/2)/WFIX)); return y,er,-tmm_h.ht_derivative(y,er),1.0
    u=y-L/2; D=u**2+WFIX**2; A=4.0
    if kind=='lorentzian': return y, EB+A*WFIX*u/D, A*WFIX**2/D, EB
    if kind=='tapered':    return y, EB+A*WFIX*u/D/np.cosh(u/(0.5*lam)), A*WFIX**2/D, EB
t0=time.time(); client=mph.start(); model=client.load(MODEL); jm=model.java
comp=jm.component('comp1'); pml=comp.coordSystem('pml1'); STUDY=model.studies()[0]
old=dict(np.load(OUT)); RES={}
NEL=sorted(set(int(round(v)) for v in np.geomspace(5,1000,22)))
jm.study('std1').feature('stat').set('plistarr',jpype.JArray(jpype.JString)([ANG]))

print('--- sKK profiles, L=8lam, sweeping elements across the terminator ---',flush=True)
for k in ['logistic','lorentzian','tapered']:
    y,er,ei,el=profile(k)
    comp.func('i_re').set('table',jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,er)]))
    comp.func('i_im').set('table',jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,ei)]))
    model.parameter('eps_low',repr(float(el))); pml.active(False)
    model.parameter('L_term','8*lam'); model.parameter('L_low','2*lam'); model.parameter('N_low','400')
    for N in NEL:
        model.parameter('N_term',str(N)); model.solve(STUDY)
        R=np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
        RES[f'P7_EL_{k}_N{N}']=np.vstack([np.array(THS)[:len(R)],R])
    print(f'  {k:11s} done  R(0)={RES[f"P7_EL_{k}_N1000"][1][0]:.3e}  R(89)={RES[f"P7_EL_{k}_N1000"][1][3]:.3e}',flush=True)

print('--- PML, 2 lam thick, sweeping elements across the PML ---',flush=True)
comp.func('i_re').set('table',jpype.JArray(jpype.JString,2)([['0',repr(EB)],[repr(2*lam),repr(EB)]]))
comp.func('i_im').set('table',jpype.JArray(jpype.JString,2)([['0','0'],[repr(2*lam),'0']]))
model.parameter('eps_low','n_bg^2'); model.parameter('L_term','1*lam'); model.parameter('N_term','400')
model.parameter('L_low','2*lam'); pml.active(True)
for pf in [1,4]:
    pml.set('PMLfactor',str(pf))
    for N in NEL:
        model.parameter('N_low',str(N)); model.solve(STUDY)
        R=np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
        RES[f'P7_PML_pf{pf}_N{N}']=np.vstack([np.array(THS)[:len(R)],R])
    print(f'  pf={pf} done  R(0)={RES[f"P7_PML_pf{pf}_N1000"][1][0]:.3e}  R(89)={RES[f"P7_PML_pf{pf}_N1000"][1][3]:.3e}',flush=True)
old.update(RES); np.savez(OUT,**old); client.clear(); print(f'done {time.time()-t0:.0f}s')
