"""Sweep (c): scale L and the feature width TOGETHER at fixed L/w.
Truncation is then constant by construction, so this isolates absolute size vs wavelength.
By scale invariance (non-dispersive media) this is equivalent to a wavelength sweep at fixed geometry.
PML thickness is scaled the same way, so its flatness is demonstrated, not assumed."""
import os, sys, time, numpy as np, jpype, mph
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,ROOT); import tmm_helper as tmm_h
MODEL=os.path.join(ROOT,'Comsol Files','skk_master.mph'); OUT=os.path.join(HERE,'master_data.npz')
lam=0.8e-6; EB=1.7**2; ANG='0 45 80 89'; THS=np.array([0.,45.,80.,89.])

def profile(kind,L,w):
    y=np.linspace(0.0,L,6001)
    if kind=='logistic':
        er=1.0+(EB-1.0)/(1.0+np.exp(-(y-L/2)/w)); return y,er,-tmm_h.ht_derivative(y,er),1.0
    u=y-L/2; D=u**2+w**2; A=4.0
    if kind=='lorentzian': return y, EB+A*w*u/D, A*w**2/D, EB
    if kind=='tapered':    return y, EB+A*w*u/D/np.cosh(u/(2.5*w)), A*w**2/D, EB
    # taper length tied to w (ell = 2.5w) so the whole shape is self-similar

t0=time.time(); client=mph.start(); model=client.load(MODEL); jm=model.java
comp=jm.component('comp1'); pml=comp.coordSystem('pml1'); STUDY=model.studies()[0]
old=dict(np.load(OUT)); RES={}; nd=[0]
jm.study('std1').feature('stat').set('plistarr',jpype.JArray(jpype.JString)([ANG]))
LG=[round(float(v),3) for v in np.geomspace(0.5,100,18)]

def push(y,er,ei):
    comp.func('i_re').set('table',jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,er)]))
    comp.func('i_im').set('table',jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,ei)]))

for ratio in [40,80]:
    print(f'\n=== L/w = {ratio} ===',flush=True)
    for k in ['logistic','lorentzian','tapered']:
        for Lr in LG:
            L=Lr*lam; w=L/ratio
            N=int(min(8000,max(20*ratio, 40*Lr)))
            y,er,ei,el=profile(k,L,w); push(y,er,ei)
            model.parameter('eps_low',repr(float(el))); pml.active(False)
            model.parameter('L_term',f'{Lr}*lam'); model.parameter('L_low','2*lam')
            model.parameter('N_term',str(N)); model.parameter('N_low','400')
            try:
                model.solve(STUDY); R=np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
            except Exception as e:
                print(f'  !! {k} r{ratio} L{Lr}: {str(e).splitlines()[0][:50]}',flush=True); continue
            RES[f'P8_r{ratio}_{k}_L{Lr}']=np.vstack([THS[:len(R)],R]); nd[0]+=1
        v=RES.get(f'P8_r{ratio}_{k}_L{LG[-1]}')
        print(f'  {k:11s} at L=100lam:  R(0)={v[1][0]:.3e}  R(80)={v[1][2]:.3e}',flush=True)

print('\n=== PML with thickness scaled the same way ===',flush=True)
push(np.array([0.0,lam]),np.array([EB,EB]),np.array([0.0,0.0]))
model.parameter('eps_low','n_bg^2'); model.parameter('L_term','1*lam')
model.parameter('N_term','400'); pml.active(True)
for pf in [1,4]:
    pml.set('PMLfactor',str(pf))
    for Lr in LG:
        model.parameter('L_low',f'{Lr}*lam')
        model.parameter('N_low',str(int(min(8000,max(400,40*Lr)))))
        model.solve(STUDY); R=np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
        RES[f'P8_PML_pf{pf}_L{Lr}']=np.vstack([THS[:len(R)],R]); nd[0]+=1
    a=RES[f'P8_PML_pf{pf}_L{LG[0]}'][1]; b=RES[f'P8_PML_pf{pf}_L{LG[-1]}'][1]
    print(f'  pf={pf}: R(80) at L=0.5lam {a[2]:.3e} -> at L=100lam {b[2]:.3e}',flush=True)

old.update(RES); np.savez(OUT,**old); client.clear()
print(f'\nDONE {nd[0]} solves in {time.time()-t0:.0f}s',flush=True)
