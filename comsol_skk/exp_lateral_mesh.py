"""C. Are 5 lateral elements across Lx = lam/4 enough at grazing incidence?
At 89 deg the lateral wavelength is lam/(n_b sin theta); N_x = 5 is ~12 elements per lateral
wavelength in air. Measure N_x in {5, 10, 20} and see whether R moves.
"""
import time, numpy as np, mph, skk_lib as S

ANG = [0, 80, 85, 89]; L = 10*S.LAM; N = 400; NXS = [5, 10, 20]
CASES = [('turned_logistic', dict(w=S.LAM/5)), ('tapered', dict(gam=S.LAM/5, ell=0.5*S.LAM, amp=4.0)),
         ('PML', dict(pf=4))]

t0 = time.time(); client = mph.start(); model = S.load(client, 'swap')
res, grid = {}, []
S.set_thickness(model, L); S.set_mesh(model, N, 'uniform')
for kind, p in CASES:
    S.set_background(model, 1.0)
    if kind == 'PML':
        S.set_profile(model, S.write_profile('flat_air', S.profile('flat', S.LAM, eps_b=1.0, npts=2)))
        S.set_pml(model, True, p['pf'])
    else:
        S.set_pml(model, False)
        S.set_profile(model, S.write_profile(f'{kind}_w5_L10lam', S.profile(kind, L, eps_b=1.0, **p)))
    for nx in NXS:
        S.set_lateral(model, nx)
        k = S.key(kind=kind, Nx=nx)
        th, R = S.solve_angles(model, ANG); res[k] = np.vstack([th, R])
        grid.append(dict(key=k, kind=kind, Nx=nx, **{a: b for a, b in p.items()}))
        print(f'  {k:28s} ' + ' '.join(f'R({a})={x:.3e}' for a, x in zip(th, R)), flush=True)
S.set_lateral(model, 5); S.set_pml(model, False)
S.save_result('lateral_mesh', res, dict(geometry='swap', pol='s', angles=ANG, L_over_lam=10,
                                        N_term=N, Nx=NXS, Lx='lam/4', background=1.0, grid=grid))
client.clear(); print(f'DONE {len(res)} configurations in {time.time()-t0:.0f}s', flush=True)
