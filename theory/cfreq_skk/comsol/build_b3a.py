"""B3a in COMSOL: a homogeneous 1-lambda single-Lorentz slab driven by a decaying wave.

Built from scratch with COMSOL's built-in "Electromagnetic Waves, Transient" interface (tag ewt in 6.4)
and its built-in Drude-Lorentz dispersion model (Wave Optics guide, pp. 194-196):

    D = eps0 eps_inf E + P,     P'' + Gamma P' + omega_n^2 P = eps0 f omega_p^2 E
    =>  eps(w) = 1 + wp^2 / (w0^2 - w^2 - i gam w)      (eps_inf = 1, f = 1)

i.e. exactly the layer model of cfreq_lib.eps_lorentz. Geometry: a thin 2-D strip (periodic
top/bottom = a plane wave at normal incidence), air | slab | air, with scattering boundaries on
both ends; the left one injects the drive. Same layout as COMSOL's own Application Library
model "Time-Domain Modeling of Dispersive Drude-Lorentz Media".

The slab is designed exactly as in B3a (cfreq_lib.lorentz_design_gamma): pick Im chi at w_c and
the damping ratio gamma/2|w''|; w0 and wp then follow in closed form (Re chi = 0 here):
    Lam = a (gam - 2b) / Im chi,   wp^2 = Lam (Im chi)^2,   w0^2 = a^2 - b^2 + gam b
which gives eps(w_c) = 1 + i Im chi exactly.

Each case is run twice, with a cos and a sin carrier (q = 0, 1), so that the complex response
E_cos - i E_sin can be compared with the complex-frequency prediction without envelope tricks.

Run:  python build_b3a.py            -> B3a_slab_transient.mph + ../data/b3a_comsol.npz
      python build_b3a.py --dt 200   (time step T/200, convergence check, no .mph saved)
"""
import os, sys, time, argparse
import numpy as np
import mph

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')

ETA = 0.05            # drive decay ratio |w''|/w'
T0, TON = 2.0, 10.0   # drive starts at T0 (entry face), Planck turn-on over TON  [periods]
TEND = 66.0           # total simulated time [periods] (= T0 + 1 slab transit + 63)
XA, L, XB = 1.0, 1.0, 1.0   # air | slab | air  [lambda]
H = 1 / 60            # strip height [lambda] (one element tall)
HMAX = 1 / 60         # mesh size [lambda]
# (Im chi at w_c, gamma/2|w''|): the solid red (virtual gain) and solid blue (loss) curves of B3a
CASES = [(-0.3, 0.5), (0.3, 1.5)]


def build(client, dt_frac=100, out_frac=16, rhoinf=0.75):
    m = client.create('B3a_slab_transient')
    j = m.java

    # ---- parameters (everything visible in Global Definitions > Parameters) ----
    P = j.param()
    for name, expr, desc in [
        ('lam0', '1[um]', 'carrier wavelength lambda_r'),
        ('T', 'lam0/c_const', 'carrier period T_r'),
        ('a', '2*pi[rad]/T', "carrier w' = w_r"),
        ('eta', f'{ETA}', "decay ratio |w''|/w'"),
        ('b', 'eta*a', "drive decay rate |w''|"),
        ('T0', f'{T0}*T', 'drive turn-on starts (at the slab entry face)'),
        ('Ton', f'{TON}*T', 'Planck turn-on duration'),
        ('q', '0', 'carrier: 0 = cos, 1 = sin'),
        ('chi_im', '-0.3', "target Im chi at w_c (eps(w_c) = 1 + i chi_im)"),
        ('ratio', '0.5', "damping ratio gamma/2|w''|"),
        ('gam', 'ratio*2*b', 'Lorentz damping gamma'),
        ('Lam', 'a*(gam-2*b)/chi_im', 'design constant (> 0)'),
        ('wp', 'sqrt(Lam*chi_im^2)', 'plasma frequency omega_p'),
        ('w0', 'sqrt(a^2-b^2+gam*b)', 'resonance omega_0'),
        ('XA', f'{XA}*lam0', 'air, left'), ('L', f'{L}*lam0', 'slab thickness'),
        ('XB', f'{XB}*lam0', 'air, right'), ('H', f'{H}*lam0', 'strip height (periodic)'),
    ]:
        P.set(name, expr, desc)

    comp = j.component().create('comp1', True)
    g = comp.geom().create('geom1', 2)
    for tag, x0, w in [('air_l', '-XA', 'XA'), ('slab', '0', 'L'), ('air_r', 'L', 'XB')]:
        r = g.create(tag, 'Rectangle'); r.set('pos', [x0, '0']); r.set('size', [w, 'H'])
    g.run()

    # ---- the drive: Planck turn-on, then e^{-b tau} (cos or sin)(a tau), tau = t - T0 - x/c ----
    V = comp.variable().create('var1')
    V.set('tau', 't-T0-x/c_const', 'retarded time (= 0 at the entry face when the drive starts)')
    V.set('uu', 'min(max(tau/Ton,1e-6),1-1e-6)', 'turn-on progress (clamped)')
    V.set('rmp', 'if(tau<=0,0,if(tau>=Ton,1,0.5*(1-tanh((1/uu-1/(1-uu))/2))))',
          'Planck (C-infinity) turn-on')
    V.set('Ein', 'rmp*exp(-b*max(tau,0))*((1-q)*cos(a*tau)+q*sin(a*tau))*1[V/m]',
          'incident field (travelling +x)')
    V.set('Ez_scaled', 'ewt.Ez*exp(b*max(t-T0,0))', 'field divided by the drive envelope')

    # ---- selections ----
    eps_ = '1e-3*lam0'
    def box(tag, dim, x0, x1, y0, y1, label):
        s = comp.selection().create(tag, 'Box'); s.label(label)
        s.set('entitydim', str(dim)); s.set('condition', 'inside')
        s.set('xmin', x0); s.set('xmax', x1); s.set('ymin', y0); s.set('ymax', y1)
    box('sel_slab', 2, f'-{eps_}', f'L+{eps_}', f'-{eps_}', f'H+{eps_}', 'Slab')
    box('sel_left', 1, f'-XA-{eps_}', f'-XA+{eps_}', f'-{eps_}', f'H+{eps_}', 'Left end (source)')
    box('sel_right', 1, f'L+XB-{eps_}', f'L+XB+{eps_}', f'-{eps_}', f'H+{eps_}', 'Right end')
    box('sel_bot', 1, f'-XA-{eps_}', f'L+XB+{eps_}', f'-{eps_}', f'{eps_}', 'Bottom edges')
    box('sel_top', 1, f'-XA-{eps_}', f'L+XB+{eps_}', f'H-{eps_}', f'H+{eps_}', 'Top edges')
    u = comp.selection().create('sel_tb', 'Union'); u.label('Top and bottom edges')
    u.set('entitydim', '1'); u.set('input', ['sel_bot', 'sel_top'])
    box('pt_entry', 0, f'-{eps_}', f'{eps_}', f'-{eps_}', f'{eps_}', 'Entry face point')
    box('pt_exit', 0, f'L-{eps_}', f'L+{eps_}', f'-{eps_}', f'{eps_}', 'Exit face point')
    for tag, sel in [('at_entry', 'pt_entry'), ('at_exit', 'pt_exit')]:
        op = comp.cpl().create(tag, 'Integration'); op.selection().geom('geom1', 0)
        op.selection().named(sel)

    # ---- physics: built-in Electromagnetic Waves, Transient ----
    ph = comp.physics().create('ewt', 'ElectromagneticWavesTransient', 'geom1')
    ph.prop('components').set('components', 'outofplane')        # E = E_z, propagation along x
    w1 = ph.feature('wee1'); w1.label('Air')
    w1.set('DisplacementFieldModel', 'RefractiveIndex'); w1.set('n_mat', 'userdef'); w1.set('n', '1')
    w2 = ph.create('wee2', 'WaveEquationElectric', 2); w2.label('Slab: Drude-Lorentz')
    w2.selection().named('sel_slab')
    w2.set('DisplacementFieldModel', 'DrudeLorentzDispersionModel')
    w2.set('epsilonInf_mat', 'userdef'); w2.set('epsilonInf', '1'); w2.set('omegap', 'wp')
    w2.set('mur_mat', 'userdef'); w2.set('mur', '1'); w2.set('sigma_mat', 'userdef'); w2.set('sigma', '0')
    w2.create('dlp1', 'DrudeLorentzPolarization', 2)
    d = w2.feature('dlp1'); d.label('Lorentz oscillator')
    d.set('item.f', '1'); d.set('item.omega0', 'w0'); d.set('item.damp', 'gam')
    s1 = ph.create('sctr1', 'Scattering', 1); s1.label('Source + absorber (left)')
    s1.selection().named('sel_left'); s1.set('IncidentField', 'EField'); s1.set('E0i', ['0', '0', 'Ein'])
    s2 = ph.create('sctr2', 'Scattering', 1); s2.label('Absorber (right)')
    s2.selection().named('sel_right')
    pc = ph.create('pc1', 'PeriodicCondition', 1); pc.label('Periodic top/bottom (plane wave)')
    pc.selection().named('sel_tb')

    # ---- mesh: structured quads ----
    me = comp.mesh().create('mesh1')
    mp = me.create('map1', 'Map')
    me.feature('size').set('custom', 'on'); me.feature('size').set('hmax', f'{HMAX}*lam0')
    me.run()

    # ---- study: time dependent, swept over (case, carrier) ----
    std = j.study().create('std1')
    tm = std.create('time', 'Transient')
    tm.set('tlist', f'range(0,T/{out_frac},{TEND}*T)')
    ps = std.create('param', 'Parametric')
    combos = [(ci, rr, qq) for ci, rr in CASES for qq in (0, 1)]
    ps.set('sweeptype', 'sparse')                                 # "specified combinations"
    ps.set('pname', ['chi_im', 'ratio', 'q'])
    ps.set('plistarr', [' '.join(str(c[k]) for c in combos) for k in range(3)])
    ps.set('punit', ['', '', ''])
    std.createAutoSequences('all')
    for s in j.sol().tags():
        for f in j.sol(s).feature().tags():
            fe = j.sol(s).feature(f)
            if fe.getType() == 'Time':
                fe.set('timemethod', 'genalpha'); fe.set('tstepsgenalpha', 'manual')
                fe.set('timestepgenalpha', f'T/{dt_frac}'); fe.set('rhoinf', str(rhoinf))
                fe.set('rtol', '1e-6')
    return m, combos


def add_plots(m):
    """Result nodes for looking at the model in the GUI."""
    j = m.java
    ds = [str(t) for t in j.result().dataset().tags()]
    dset = [t for t in ds if j.result().dataset(t).getType() == 'Solution'][-1]
    # 1) probe time traces: field at the exit face and the drive, and the ratio
    pg = j.result().create('pg_trace', 'PlotGroup1D'); pg.label('Exit-face field vs time')
    pg.set('data', dset)
    gl = pg.create('glob1', 'Global'); gl.set('expr', ['at_exit(ewt.Ez)', 'at_entry(Ein)'])
    gl.set('descr', ['transmitted field at exit face', 'drive at entry face'])
    # 2) space-time map of the field divided by the drive envelope
    pg2 = j.result().create('pg_x', 'PlotGroup1D'); pg2.label('E_z / drive envelope along the strip')
    pg2.set('data', dset)
    j.result().dataset().create('cln', 'CutLine2D')
    cl = j.result().dataset('cln'); cl.set('data', dset)
    cl.set('genpoints', [['-XA', 'H/2'], ['L+XB', 'H/2']])
    pg2.set('data', 'cln')
    ln = pg2.create('lngr1', 'LineGraph'); ln.set('expr', 'Ez_scaled'); ln.set('xdata', 'expr')
    ln.set('xdataexpr', 'x/lam0')
    pg3 = j.result().create('pg_2d', 'PlotGroup2D'); pg3.label('E_z in the strip')
    pg3.set('data', dset); pg3.create('surf1', 'Surface').set('expr', 'ewt.Ez')


def extract(m, combos):
    j = m.java
    ds = [str(t) for t in j.result().dataset().tags()]
    labels = {t: str(j.result().dataset(t).label()) for t in ds}
    dset = [t for t in ds if j.result().dataset(t).getType() == 'Solution'][-1]
    print('mph datasets:', [str(d) for d in m.datasets()], 'java label:', labels[dset])
    node = [d for d in m.datasets() if 'Parametric' in str(d)][0]
    print('using', node)
    out = []
    for k in range(len(combos)):
        r = np.asarray(m.evaluate(['t', 'at_exit(ewt.Ez)', 'at_entry(ewt.Ez)'],
                                  dataset=node, outer=k + 1)).squeeze()
        if r.shape[0] != 3:
            r = r.T
        t, e_exit, e_in = r
        out.append((t, e_exit, e_in))
    return out, labels[dset]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dt', type=int, default=100, help='time step = T/dt')
    ap.add_argument('--rhoinf', type=float, default=0.75)
    ap.add_argument('--nosave', action='store_true')
    A = ap.parse_args()
    t0 = time.time()
    client = mph.start()
    m, combos = build(client, dt_frac=A.dt, rhoinf=A.rhoinf)
    print(f'built in {time.time()-t0:.0f}s; solving {len(combos)} runs ...'); t1 = time.time()
    m.java.study('std1').run()
    print(f'solved in {time.time()-t1:.0f}s')
    res, dlabel = extract(m, combos)
    T = 1e-6 / 299792458.0
    tag = f'dt{A.dt}_rho{A.rhoinf:g}'
    np.savez(os.path.join(DATA, f'b3a_comsol_{tag}.npz'),
             t=res[0][0] / T, combos=np.array(combos, float),
             e_exit=np.array([r[1] for r in res]), e_entry=np.array([r[2] for r in res]),
             eta=ETA, T0=T0, TON=TON, L=L)
    print('saved data', tag, 'from dataset', dlabel)
    if not A.nosave:
        add_plots(m)
        m.save(os.path.join(HERE, 'B3a_slab_transient.mph'))
        print('saved B3a_slab_transient.mph')
