"""R vs incidence angle for every terminator, at identical thickness AND identical mesh.

This is the template every other exp_*.py is copied from. It is also the experiment that
the `swap` geometry exists for: the PML and the sKK profiles occupy the SAME domain, so
they share L_term and N_term and the comparison is equal-footprint and equal-cost by
construction rather than by argument.

    background n_b = 1.7  |  TERMINATOR (L_term, N_term)  |  PEC

Run:    python exp_angle_vs_terminator.py
Then:   python fig_angle_vs_terminator.py
"""
import time
import numpy as np, mph
import skk_lib as S

# ---------------------------------------------------------------- configuration
NB      = 1.7
EB      = NB**2
L       = 10*S.LAM          # terminator thickness, identical for every curve
N_ELEM  = 800               # elements across the terminator, identical for every curve
W       = S.LAM/5           # feature width, fixed in absolute units
ELL     = 0.5*S.LAM         # sech taper length for the tapered Lorentzian
AMP     = 4.0               # Lorentzian amplitude
ANGLES  = np.arange(0, 90, 1.0)
PFS     = [1, 4]            # PMLfactors to measure against

KINDS = {'logistic':        dict(w=W),
         'lorentzian':      dict(gam=W, amp=AMP),
         'tapered':         dict(gam=W, ell=ELL, amp=AMP),
         'turned_logistic': dict(w=W)}

# ---------------------------------------------------------------- run
t0 = time.time()
client = mph.start()
model = S.load(client, 'swap')
model.parameter('n_bg', str(NB))
model.parameter('L_term', f'{L/S.LAM:g}*lam')
S.set_mesh(model, N_ELEM, 'uniform')

results, meta = {}, dict(geometry='swap', pol='s', n_bg=NB, L_over_lam=L/S.LAM,
                         N_term=N_ELEM, mesh='uniform', angles_deg=[0, 89, 1],
                         profiles={}, note='identical thickness and mesh on every curve; '
                                           'PML occupies the same domain as the sKK profiles')

print('=== sKK profiles ===', flush=True)
S.set_pml(model, False)
for kind, params in KINDS.items():
    prof = S.profile(kind, L, eps_b=EB, **params)
    path = S.write_profile(f'{kind}_L{L/S.LAM:g}lam', prof)
    S.set_profile(model, path)
    th, R = S.solve_angles(model, ANGLES)
    results[f'R_{kind}'] = np.vstack([th, R])
    meta['profiles'][kind] = dict(prof.meta, file=path.replace(S.HERE + '/', ''),
                                  passive=prof.passive(),
                                  one_sidedness_pct=round(prof.one_sidedness(), 4))
    print(f'  {kind:16s} R(0)={R[0]:.3e} R(45)={R[45]:.3e} R(80)={R[80]:.3e} '
          f'R(89)={R[89]:.3e}  [{time.time()-t0:.0f}s]', flush=True)

print('=== PML, same domain, same thickness, same mesh ===', flush=True)
flat = S.write_profile('flat_background', S.profile('flat', L, eps_b=EB, npts=2))
S.set_profile(model, flat)
for pf in PFS:
    S.set_pml(model, True, pf)
    th, R = S.solve_angles(model, ANGLES)
    results[f'R_PML_pf{pf}'] = np.vstack([th, R])
    print(f'  PML pf={pf:<12d} R(0)={R[0]:.3e} R(45)={R[45]:.3e} R(80)={R[80]:.3e} '
          f'R(89)={R[89]:.3e}  [{time.time()-t0:.0f}s]', flush=True)
meta['pml_factors'] = PFS

S.save_result('angle_vs_terminator', results, meta)
client.clear()
print(f'DONE {len(results)} solves in {time.time()-t0:.0f}s', flush=True)
