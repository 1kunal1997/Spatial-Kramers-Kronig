"""D. Discrete Hilbert transform ON THE MESH NODES vs the continuous pair sampled there.

The table is placed at the N+1 node positions of a uniform mesh, so the profile COMSOL sees
is the piecewise-linear interpolant of the node sequence in BOTH modes. Only the partner
differs:
    continuous   fine-grid Hilbert pair, sampled at the nodes
    dht          the driving part sampled at the nodes, then the DHT of that sequence
Logistic keeps n_bg = 1.7 (it needs the index step); turned logistic is in air.
"""
import time, numpy as np, mph, skk_lib as S

ANG = [0, 80, 89]; L = 10*S.LAM; W = S.LAM/5; NS = [50, 100, 200]
KINDS = ['logistic', 'turned_logistic']; MODES = ['continuous', 'dht']

t0 = time.time(); client = mph.start(); model = S.load(client, 'swap')
res, grid = {}, []
S.set_pml(model, False); S.set_thickness(model, L)
for kind in KINDS:
    nb = S.background(kind); S.set_background(model, nb)
    for N in NS:
        S.set_mesh(model, N, 'uniform')
        for mode in MODES:
            prof = S.profile_on_nodes(kind, L, N, mode, eps_b=nb**2, w=W)
            pref = S.write_profile(f'{kind}_nodes{N}_{mode}', prof); S.set_profile(model, pref)
            k = S.key(kind=kind, N=N, mode=mode)
            th, R = S.solve_angles(model, ANG); res[k] = np.vstack([th, R])
            fom = prof.one_sidedness()
            grid.append(dict(key=k, kind=kind, N=N, mode=mode, profile=pref, n_bg=nb,
                             one_sidedness=round(fom, 4)))
            print(f'  {k:44s} FoM={fom:8.4f}%  ' + ' '.join(f'R({a})={x:.2e}' for a, x in zip(th, R)),
                  flush=True)
S.save_result('discrete_hilbert', res, dict(
    geometry='swap', pol='s', angles=ANG, L_over_lam=10, w_over_lam=0.2, N=NS, kinds=KINDS,
    modes=MODES, held_fixed='L, w; table on the mesh nodes',
    background={k: S.background(k) for k in KINDS}, grid=grid))
client.clear(); print(f'DONE {len(res)} configurations in {time.time()-t0:.0f}s', flush=True)
