"""F. Turned logistic IN FRONT OF a PML (stacked model) -- the advisor's question.

Total footprint fixed at L_tot = 10 lam and element size fixed at lam/40 (400 per 10 lam),
so every configuration has the same cost. The footprint is split L_skk + L_pml = L_tot.

    reference   sKK alone (swap, 10 lam)           PML alone pf=1,4 (swap, 10 lam)
    stacked     sKK L_skk | lower region L_pml | PEC, for L_skk in {2, 5, 8} lam, with the
                lower region being:
                  'lossy+PML'   eps = 1 + i*amp -- CONTINUES the turned logistic's end value,
                                so the PML stretches the same medium; no eps'' jump. pf = 1, 4
                  'lossy'       same medium, PML OFF -- what a plain lossy slab does
                  'air+PML'     eps = 1 (lossless), pf = 4 -- the naive stack; shows the
                                cost of the eps'' jump at the sKK|PML interface
Parameters from exp A: w = lam/5, amp = 1, air background.
"""
import time, numpy as np, mph, skk_lib as S

ANG = [0, 80, 89]; LTOT = 10.0; W = S.LAM/5; AMP = 1.0; PER_LAM = 40
SPLITS = [2, 5, 8]

t0 = time.time(); client = mph.start(); res, grid = {}, []

def rec(k, th, R, **info):
    res[k] = np.vstack([th, R]); grid.append(dict(key=k, **info))
    print(f'  {k:46s} ' + ' '.join(f'R({a})={x:.2e}' for a, x in zip(th, R)) + f'  [{time.time()-t0:.0f}s]',
          flush=True)

print('=== references (swap model, 10 lam) ===', flush=True)
m = S.load(client, 'swap'); S.set_background(m, 1.0); S.set_thickness(m, LTOT*S.LAM)
S.set_mesh(m, int(PER_LAM*LTOT), 'uniform')
prof = S.profile('turned_logistic', LTOT*S.LAM, eps_b=1.0, w=W, amp=AMP)
S.set_profile(m, S.write_profile('turned_logistic_wfix_L10lam', prof)); S.set_pml(m, False)
th, R = S.solve_angles(m, ANG); rec(S.key(case='skk_alone', Lskk=LTOT), th, R, case='skk_alone', L_skk=LTOT)
S.set_profile(m, S.write_profile('flat_air', S.profile('flat', S.LAM, eps_b=1.0, npts=2)))
for pf in (1, 4):
    S.set_pml(m, True, pf); th, R = S.solve_angles(m, ANG)
    rec(S.key(case='pml_alone', pf=pf), th, R, case='pml_alone', pf=pf, L_pml=LTOT)
S.set_pml(m, False)

print('=== stacked model ===', flush=True)
m = S.load(client, 'stacked'); S.set_background(m, 1.0)
for Ls in SPLITS:
    Lp = LTOT - Ls
    prof = S.profile('turned_logistic', Ls*S.LAM, eps_b=1.0, w=W, amp=AMP)
    S.set_profile(m, S.write_profile(f'turned_logistic_wfix_L{Ls:g}lam', prof))
    S.set_thickness(m, Ls*S.LAM); S.set_mesh(m, int(PER_LAM*Ls), 'uniform')
    for lower, eps_im, pml_on, pfs in [('lossy+PML', AMP, True, (1, 4)), ('lossy', AMP, False, (0,)),
                                        ('air+PML', 0.0, True, (4,))]:
        S.set_lower(m, Lp*S.LAM, int(PER_LAM*Lp), eps_re=1.0, eps_im=eps_im)
        for pf in pfs:
            S.set_pml(m, pml_on, pf if pml_on else 1)
            th, R = S.solve_angles(m, ANG)
            rec(S.key(case=lower, Lskk=Ls, pf=pf), th, R, case=lower, L_skk=Ls, L_pml=Lp, pf=pf,
                lower_eps=[1.0, eps_im], pml=pml_on)
            if Ls == 5 and lower == 'lossy+PML' and pf == 4:
                S.snapshot(m, 'exp_stacked__turned5lam_lossyPML4_5lam')

S.save_result('stacked', res, dict(
    geometry='stacked (+swap references)', pol='s', angles=ANG, L_total=LTOT, splits_L_skk=SPLITS,
    elements_per_lam=PER_LAM, profile=dict(kind='turned_logistic', w_over_lam=0.2, amp=AMP),
    background=1.0, held_fixed='total footprint and element size', grid=grid))
client.clear(); print(f'DONE {len(res)} configurations in {time.time()-t0:.0f}s', flush=True)
