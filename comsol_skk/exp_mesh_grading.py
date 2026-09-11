"""B. Non-uniform meshes, done at a budget where the wave is resolved and at two feature
widths chosen so the predicted optimal centre/tail ratio r* ~ 0.4 lam/(n w) is ~1 at one
and ~12 at the other. If centre-dense wins at lam/50 and loses at lam/5, the two-resolution
picture in PLAN.md is confirmed.

    gradings: uniform | centre (symmetric+reverse) | entry (reverse)
    L = 10 lam, N in {100, 200, 400}, r in {1.5, 2, 3, 5, 10}, w in {lam/5, lam/50}
Background: air (BACKGROUND RULE).
"""
import time, numpy as np, mph, skk_lib as S

ANG = [0, 80, 89]; L = 10*S.LAM
NS = [100, 200, 400]; RS = [1.5, 2, 3, 5, 10]; WS = {'w5': S.LAM/5, 'w50': S.LAM/50}
KINDS = ['turned_logistic', 'tapered']

def params(kind, w):
    return dict(w=w) if kind == 'turned_logistic' else dict(gam=w, ell=2.5*w, amp=4.0)

t0 = time.time(); client = mph.start(); model = S.load(client, 'swap')
res, grid = {}, []
S.set_pml(model, False); S.set_thickness(model, L)
for kind in KINDS:
    S.set_background(model, S.background(kind)); eb = S.background(kind)**2
    for wtag, w in WS.items():
        prof = S.profile(kind, L, eps_b=eb, **params(kind, w))
        pref = S.write_profile(f'{kind}_{wtag}_L10lam', prof); S.set_profile(model, pref)
        for N in NS:
            for grading, r in [('uniform', 1)] + [(g, r) for r in RS for g in ('centre', 'entry')]:
                S.set_mesh(model, N, grading, r)
                k = S.key(kind=kind, w=wtag, N=N, grading=grading, r=r)
                th, R = S.solve_angles(model, ANG); res[k] = np.vstack([th, R])
                grid.append(dict(key=k, kind=kind, w=wtag, w_over_lam=w/S.LAM, N=N,
                                 grading=grading, r=r, profile=pref))
                print(f'  {k:60s} ' + ' '.join(f'R({a})={x:.2e}' for a, x in zip(th, R))
                      + f'  [{time.time()-t0:.0f}s]', flush=True)
S.set_mesh(model, 400, 'uniform')
S.save_result('mesh_grading', res, dict(
    geometry='swap', pol='s', angles=ANG, L_over_lam=10, N=NS, ratios=RS,
    widths={k: v/S.LAM for k, v in WS.items()}, kinds=KINDS, held_fixed='L, N; vary r and where',
    r_star_prediction={k: round(0.4/(1.0*v/S.LAM), 1) for k, v in WS.items()},
    background={k: S.background(k) for k in KINDS}, grid=grid))
client.clear(); print(f'DONE {len(res)} configurations in {time.time()-t0:.0f}s', flush=True)
