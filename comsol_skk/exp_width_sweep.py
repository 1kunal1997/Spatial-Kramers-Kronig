"""E. Shape parameters one at a time, at fixed L = 10 lam and N = 400 (resolved).
    w/lam in {0.05, 0.1, 0.2, 0.5, 1}  for turned logistic, tapered and plain Lorentzian
    ell/lam in {0.25, 0.5, 1, 2}       tapered, gam = lam/5
    A in {1, 2, 4, 8}                   tapered and Lorentzian, gam = lam/5
    amp in {0.5, 1, 2}                  turned logistic, w = lam/5
Background: air (BACKGROUND RULE). The one-sidedness FoM is recorded for every profile.
"""
import time, numpy as np, mph, skk_lib as S

ANG = [0, 80, 89]; L = 10*S.LAM; N = 400
RUNS = []
for wl in [0.05, 0.1, 0.2, 0.5, 1]:
    RUNS += [('w', wl, 'turned_logistic', dict(w=wl*S.LAM)),
             ('w', wl, 'tapered', dict(gam=wl*S.LAM, ell=2.5*wl*S.LAM, amp=4.0)),
             ('w', wl, 'lorentzian', dict(gam=wl*S.LAM, amp=4.0))]
for el in [0.25, 0.5, 1, 2]:
    RUNS.append(('ell', el, 'tapered', dict(gam=S.LAM/5, ell=el*S.LAM, amp=4.0)))
for A in [1, 2, 4, 8]:
    RUNS += [('A', A, 'tapered', dict(gam=S.LAM/5, ell=0.5*S.LAM, amp=float(A))),
             ('A', A, 'lorentzian', dict(gam=S.LAM/5, amp=float(A)))]
for am in [0.5, 1, 2]:
    RUNS.append(('amp', am, 'turned_logistic', dict(w=S.LAM/5, amp=am)))

t0 = time.time(); client = mph.start(); model = S.load(client, 'swap')
res, grid = {}, []
S.set_pml(model, False); S.set_thickness(model, L); S.set_mesh(model, N, 'uniform')
for sweep, val, kind, p in RUNS:
    S.set_background(model, S.background(kind)); eb = S.background(kind)**2
    prof = S.profile(kind, L, eps_b=eb, **p)
    pref = S.write_profile(f'{kind}_{sweep}{val:g}_L10lam', prof); S.set_profile(model, pref)
    k = S.key(kind=kind, sweep=sweep, val=val)
    th, R = S.solve_angles(model, ANG); res[k] = np.vstack([th, R]); fom = prof.one_sidedness()
    grid.append(dict(key=k, kind=kind, sweep=sweep, val=val, profile=pref,
                     one_sidedness=round(fom, 3), eps_re_min=round(float(prof.eps_re.min()), 3),
                     **{a: (b/S.LAM if a in ('w', 'gam', 'ell') else b) for a, b in p.items()}))
    print(f'  {k:40s} FoM={fom:7.3f}%  ' + ' '.join(f'R({a})={x:.2e}' for a, x in zip(th, R))
          + f'  [{time.time()-t0:.0f}s]', flush=True)
S.save_result('width_sweep', res, dict(
    geometry='swap', pol='s', angles=ANG, L_over_lam=10, N=N, held_fixed='L, N',
    background={k: S.background(k) for k in ['turned_logistic', 'tapered', 'lorentzian']}, grid=grid))
client.clear(); print(f'DONE {len(res)} configurations in {time.time()-t0:.0f}s', flush=True)
