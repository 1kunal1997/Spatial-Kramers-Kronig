"""Save one representative, fully-configured model per experiment to Models/snapshots/.
NO solves. Open one in the COMSOL GUI, press Compute, and compare with the number stored
under the same key in data/<experiment>.npz (print it with:  python -c "import skk_lib as S;
r,m=S.load_result('budget_vs_length'); print(r['kind=turned_logistic__rule=wfix__L=10__N=400'])").

The template models in Models/ are never touched. exp_stacked.py writes its own snapshot.
"""
import mph, skk_lib as S

L10 = 10*S.LAM; W = S.LAM/5
client = mph.start()

def fresh(geometry='swap', nb=1.0, L=L10, N=400):
    m = S.load(client, geometry); S.set_background(m, nb); S.set_thickness(m, L)
    S.set_mesh(m, N, 'uniform'); S.set_pml(m, False); return m

# angle_vs_terminator: the one profile NOT in air -- the paper's logistic at n_b = 1.7
m = fresh(nb=1.7, N=800)
S.set_profile(m, S.write_profile('logistic_L10lam', S.profile('logistic', L10, eps_b=1.7**2, w=W)))
S.snapshot(m, 'exp_angle_vs_terminator__logistic_nb1.7_L10_N800')

# A. budget_vs_length: key = kind=turned_logistic__rule=wfix__L=10__N=400, and its PML opponent
m = fresh()
S.set_profile(m, S.write_profile('turned_logistic_wfix_L10lam',
                                 S.profile('turned_logistic', L10, eps_b=1.0, w=W)))
S.snapshot(m, 'exp_budget_vs_length__turned_logistic_wfix_L10_N400')
S.set_profile(m, S.write_profile('flat_air', S.profile('flat', S.LAM, eps_b=1.0, npts=2)))
S.set_pml(m, True, 4); S.snapshot(m, 'exp_budget_vs_length__PML_pf4_L10_N400')

# B. mesh_grading: key = kind=turned_logistic__w=w5__N=200__grading=centre__r=3
m = fresh(N=200); S.set_mesh(m, 200, 'centre', 3)
S.set_profile(m, S.write_profile('turned_logistic_w5_L10lam',
                                 S.profile('turned_logistic', L10, eps_b=1.0, w=W)))
S.snapshot(m, 'exp_mesh_grading__turned_logistic_w5_N200_centre_r3')

# C. lateral_mesh: key = kind=turned_logistic__Nx=20
m = fresh(); S.set_lateral(m, 20)
S.set_profile(m, S.write_profile('turned_logistic_w5_L10lam',
                                 S.profile('turned_logistic', L10, eps_b=1.0, w=W)))
S.snapshot(m, 'exp_lateral_mesh__turned_logistic_Nx20')

# D. discrete_hilbert: key = kind=turned_logistic__N=100__mode=dht
m = fresh(N=100)
S.set_profile(m, S.write_profile('turned_logistic_nodes100_dht',
                                 S.profile_on_nodes('turned_logistic', L10, 100, 'dht', eps_b=1.0, w=W)))
S.snapshot(m, 'exp_discrete_hilbert__turned_logistic_N100_dht')

# E. width_sweep: key = kind=tapered__sweep=w__val=0.2
m = fresh()
S.set_profile(m, S.write_profile('tapered_w0.2_L10lam',
                                 S.profile('tapered', L10, eps_b=1.0, gam=W, ell=0.5*S.LAM, amp=4.0)))
S.snapshot(m, 'exp_width_sweep__tapered_w0.2_L10_N400')

client.clear(); print('done')
