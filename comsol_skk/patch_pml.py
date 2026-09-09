"""Re-run the PML sections with a finely-meshed companion domain so the measured
floor is the PML's, not the uniform-region under-resolution."""
import os, sys, time, numpy as np, jpype, mph
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE=os.path.dirname(os.path.abspath(__file__))
MODEL=os.path.join(ROOT,'Comsol Files','skk_master.mph'); OUT=os.path.join(HERE,'master_data.npz')
lam=0.8e-6; EB=1.7**2
FINE='range(0,1,89)'; COARSE='0 10 20 30 40 50 60 70 80 89'
t0=time.time()
client=mph.start(); model=client.load(MODEL); jm=model.java; comp=jm.component('comp1')
pml=comp.coordSystem('pml1'); STUDY=model.studies()[0]
old=dict(np.load(OUT)); RES={}

def push(y,er,ei):
    comp.func('i_re').set('table', jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,er)]))
    comp.func('i_im').set('table', jpype.JArray(jpype.JString,2)([[repr(float(a)),repr(float(b))] for a,b in zip(y,ei)]))

def solve(key, L_low, pf, N_low, fine=False, L_term=1*lam, N_term=400):
    jm.study('std1').feature('stat').set('plistarr', jpype.JArray(jpype.JString)([FINE if fine else COARSE]))
    model.parameter('L_term', f'{L_term/lam}*lam'); model.parameter('L_low', f'{L_low/lam}*lam')
    model.parameter('N_term', str(int(N_term))); model.parameter('N_low', str(int(N_low)))
    model.parameter('eps_low','n_bg^2')
    push(np.array([0.0,L_term]), np.array([EB,EB]), np.array([0.0,0.0]))
    pml.active(True); pml.set('PMLfactor', str(pf))
    model.solve(STUDY)
    R=np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
    th=np.linspace(0,89,90) if fine else np.array([0,10,20,30,40,50,60,70,80,89],float)
    RES[key]=np.vstack([th[:len(R)],R])
    RES[key+'__meta']=np.array([N_low, L_low/lam, pf, N_term],float)
    print(f'  {key:26s} N_pml={N_low:5d} R(0)={R[0]:.3e} R(89)={R[-1]:.3f}  [{time.time()-t0:.0f}s]', flush=True)

print('=== S1 PML angle curves (fine, well-resolved) ===', flush=True)
for pf in [1,2,4]: solve(f'PML_pf{pf}', 2*lam, pf, 800, fine=True)
print('\n=== S2 PML element sweep ===', flush=True)
for pf in [1,2,4]:
    for N in [20,50,100,200,400,800,1600]: solve(f'PMLN_pf{pf}_N{N}', 2*lam, pf, N)
print('\n=== S3 PML thickness sweep ===', flush=True)
for Lm in [1,2,4,8,16]: solve(f'PMLL_L{Lm}', Lm*lam, 1, 400)

old.update(RES); np.savez(OUT, **old); client.clear()
print(f'\npatched {len(RES)//2} entries in {time.time()-t0:.0f}s', flush=True)
