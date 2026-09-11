"""The 90-degree-turned logistic, in an AIR background.

eps'' is the logistic (0 at the entry face -> A deep inside: lossless -> lossy) and eps' is
its Hilbert partner, shifted so that eps' = 1 (air) at both ends.  So the terminator is
index-matched AND lossless at the entry face, and saturates at constant loss at the PEC --
the opposite assignment to the standard sKK logistic, where eps' carries the ramp.

Analytically this is the standard logistic pair multiplied by e^{-i pi/2}: a constant phase
rotation of a one-sided function is still one-sided, so the sKK property is untouched
(spectral one-sidedness FoM = 100.000% for the whole rotation family).  What changes is that
the profile now returns to AIR rather than stepping down to it, so the background is vacuum
-- a PML's home turf and the case a PML user actually has.

Sign branch fixed by the four-way control (TMM, L=10lam):
    eps' = 1 + HT[eps'']  forward   R(0) = 2.5e-09   <-- correct
    eps' = 1 + HT[eps'']  reversed  R(0) = 4.7e-02
    eps' = 1 - HT[eps'']  forward   R(0) = 5.3e-02   (eps' dips to 0.08: metallic + TIR)
    eps' = 1 - HT[eps'']  reversed  R(0) = 4.7e-02
The control is re-run here in COMSOL.  Writes P14_* keys.

The harness is the same .mph, with n_bg switched to 1 in memory only -- the model on disk is
never re-saved, so every earlier n_b = 1.7 result stands.  PML references are re-measured in
air, since the n_b = 1.7 PML numbers are not comparable.
"""
import os, sys, time, numpy as np, jpype, mph
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); import tmm_helper as tmm_h
MODEL = os.path.join(HERE, 'skk_master.mph')
OUT   = os.path.join(HERE, 'master_data.npz')

lam = 0.8e-6
FINE = 'range(0,1,89)'; TH = np.arange(0, 90, 1.0)
COARSE = '0 10 20 30 40 50 60 70 80 89'; THC = np.array([0,10,20,30,40,50,60,70,80,89], float)
W = lam/5; AMP = 1.0                       # eps'' ramps 0 -> 1 over a width lam/5
LENGTHS = [2, 5, 10, 32, 100]


def air_flip(L, s=+1, A=AMP, w=W):
    """y=L is the ENTRY face (air side), y=0 is the PEC side."""
    y = np.linspace(0.0, L, 6001)
    ei = A/(1.0 + np.exp((y - L/2)/w))          # 0 at entry, A at the PEC end
    er = 1.0 + s*tmm_h.ht_derivative(y, ei)     # ht_derivative detrends to 0 at both ends,
    return y, er, ei                            # so eps' = 1 (air) at both ends by construction


t0 = time.time(); client = mph.start(); model = client.load(MODEL); jm = model.java
comp = jm.component('comp1'); pml = comp.coordSystem('pml1'); STUDY = model.studies()[0]
old = dict(np.load(OUT)); RES = {}
model.parameter('n_bg', '1')                    # AIR background (in memory only)
model.parameter('eps_low', '1')
model.parameter('L_low', '2*lam'); model.parameter('N_low', '400')


def push(y, er, ei):
    comp.func('i_re').set('table', jpype.JArray(jpype.JString, 2)(
        [[repr(float(a)), repr(float(b))] for a, b in zip(y, er)]))
    comp.func('i_im').set('table', jpype.JArray(jpype.JString, 2)(
        [[repr(float(a)), repr(float(b))] for a, b in zip(y, ei)]))


def run(key, fine=True):
    jm.study('std1').feature('stat').set('plistarr',
        jpype.JArray(jpype.JString)([FINE if fine else COARSE]))
    th = TH if fine else THC
    try:
        model.solve(STUDY)
        R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
    except Exception as e:
        print(f'  !! {key}: {str(e).splitlines()[0][:70]}', flush=True); return
    RES[key] = np.vstack([th[:len(R)], R])
    i80 = int(np.argmin(abs(th-80)))
    print(f'  {key:26s} R(0)={R[0]:.3e} R(80)={R[i80]:.3e} R(89)={R[-1]:.3e} '
          f' [{time.time()-t0:.0f}s]', flush=True)


print('=== A: four-way sign/orientation control, L = 10 lam ===', flush=True)
pml.active(False)
model.parameter('L_term', '10*lam'); model.parameter('N_term', '500')
for s in (+1, -1):
    y, er, ei = air_flip(10*lam, s)
    for rev, tag in ((False, 'fwd'), (True, 'rev')):
        push(y, er[::-1] if rev else er, ei[::-1] if rev else ei)
        run(f'P14_CTRL_s{"p" if s>0 else "m"}_{tag}', fine=False)

print('=== B: PML reference IN AIR ===', flush=True)
push(np.array([0.0, lam]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
model.parameter('L_term', '1*lam'); model.parameter('N_term', '400')
model.parameter('N_low', '800'); pml.active(True)
for pf in (1, 4):
    pml.set('PMLfactor', str(pf)); run(f'P14_PML_pf{pf}')

print('=== C: turned logistic in air, thickness sweep ===', flush=True)
pml.active(False); model.parameter('N_low', '400')
for L in LENGTHS:
    N = int(min(8000, max(400, 50*L)))
    y, er, ei = air_flip(L*lam)
    push(y, er, ei)
    model.parameter('L_term', f'{L}*lam'); model.parameter('N_term', str(N))
    run(f'P14_AIR_L{L}')
    if L == 10:
        RES['P14_PROF'] = np.vstack([y/lam, er, ei])

old.update(RES); np.savez(OUT, **old); client.clear()
print(f'DONE {len([k for k in RES if not k.endswith("PROF")])} solves in '
      f'{time.time()-t0:.0f}s', flush=True)
