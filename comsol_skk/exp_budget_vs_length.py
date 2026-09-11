"""A. Fixed element budget: does it pay to spend elements on length or on sampling?

Fix N. Sweep L. Small L -> fine mesh, heavy truncation. Large L -> coarse mesh, unresolved
wave. The envelope of R over L at each N is the true equal-cost curve of an sKK terminator,
measured against a PML of the SAME thickness and element count (swap model).

Two width rules, kept distinct by name:
    wfix   w = lam/5     elements per feature AND per lam both fall as L grows
    wself  w = L/40      shape identical at every L; only elements per lam changes
Tapered Lorentzian uses gam = w, ell = 2.5*gam (both old conventions coincide at lam/5).
L/lam in {0.2, 0.5, 1, 2} is the paper's d/lam window (PLAN item 6).
Background: air for every profile here (BACKGROUND RULE in skk_lib).
"""
import time, numpy as np, mph, skk_lib as S

ANG   = [0, 80, 89]
NS    = [100, 200, 400]
LS    = [0.2, 0.5, 1, 2, 5, 10, 20, 50]
KINDS = ['turned_logistic', 'tapered']
PFS   = [1, 4]

def params(kind, rule, L):
    w = S.LAM/5 if rule == 'wfix' else L*S.LAM/40      # L arrives in units of lam
    return dict(w=w) if kind == 'turned_logistic' else dict(gam=w, ell=2.5*w, amp=4.0)

t0 = time.time(); client = mph.start(); model = S.load(client, 'swap')
res, grid = {}, []
S.set_pml(model, False)
for kind in KINDS:
    S.set_background(model, S.background(kind)); eb = S.background(kind)**2
    for rule in ['wfix', 'wself']:
        for L in LS:
            prof = S.profile(kind, L*S.LAM, eps_b=eb, **params(kind, rule, L))
            pref = S.write_profile(f'{kind}_{rule}_L{L:g}lam', prof)
            S.set_profile(model, pref); S.set_thickness(model, L*S.LAM)
            for N in NS:
                S.set_mesh(model, N, 'uniform')
                k = S.key(kind=kind, rule=rule, L=L, N=N)
                th, R = S.solve_angles(model, ANG); res[k] = np.vstack([th, R])
                grid.append(dict(key=k, kind=kind, rule=rule, L=L, N=N, profile=pref,
                                 one_sidedness=round(prof.one_sidedness(), 3)))
                print(f'  {k:52s} ' + ' '.join(f'R({a})={r:.2e}' for a, r in zip(th, R))
                      + f'  [{time.time()-t0:.0f}s]', flush=True)

print('=== PML in air, same thickness, same N ===', flush=True)
S.set_background(model, 1.0)
S.set_profile(model, S.write_profile('flat_air', S.profile('flat', S.LAM, eps_b=1.0, npts=2)))
for pf in PFS:
    S.set_pml(model, True, pf)
    for L in LS:
        S.set_thickness(model, L*S.LAM)
        for N in NS:
            S.set_mesh(model, N, 'uniform')
            k = S.key(kind='PML', pf=pf, L=L, N=N)
            th, R = S.solve_angles(model, ANG); res[k] = np.vstack([th, R])
            grid.append(dict(key=k, kind='PML', pf=pf, L=L, N=N))
            print(f'  {k:52s} ' + ' '.join(f'R({a})={r:.2e}' for a, r in zip(th, R)), flush=True)
S.set_pml(model, False)

S.save_result('budget_vs_length', res, dict(
    geometry='swap', pol='s', angles=ANG, N=NS, L_over_lam=LS, kinds=KINDS, pml_factors=PFS,
    width_rules={'wfix': 'w = lam/5', 'wself': 'w = L/40'}, held_fixed='N (per curve)',
    background={k: S.background(k) for k in KINDS + ['flat']}, mesh='uniform', grid=grid))
client.clear(); print(f'DONE {len(res)} configurations in {time.time()-t0:.0f}s', flush=True)
