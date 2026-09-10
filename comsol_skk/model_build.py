"""Build the template COMSOL models for the sKK-as-PML study. NO SOLVES.

    python model_build.py                 # builds both geometries, s-pol
    python model_build.py --geometry swap
    python model_build.py --pol p

Two geometries, one code path, so they cannot drift apart:

  swap    (2 domains -- the workhorse)          stacked (3 domains -- back-to-back only)
      ---- periodic Port ----                       ---- periodic Port ----
      |   background  n_bg   | L_bg                 |   background  n_bg   | L_bg
      |   TERMINATOR         | L_term               |   TERMINATOR         | L_term
      -------- PEC ----------                       |   lower region       | L_low
                                                    -------- PEC ----------
  The PML coordinate system lives on the TERMINATOR domain, so a PML run and an sKK run
  occupy the same domain and share L_term as their thickness. That is what makes an
  equal-footprint comparison structural instead of something a caption has to argue.
  A PML run just activates pml1 and loads the 'flat' profile (eps = n_bg^2).

  In `stacked` the PML lives in the lower region instead, for sKK-in-front-of-a-PML tests.

The terminator's permittivity always comes from ONE interpolation function `i_prof`, read
from a file in profiles/ (columns: y, eps_re, eps_im). Experiments swap the file; the model
never carries profile data of its own.
"""
import os, sys, argparse, time
import numpy as np, jpype, mph

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skk_lib as S

MODELS = S.MODELS


def build(client, geometry='swap', pol='s', save=True):
    stacked = (geometry == 'stacked')
    tag = f'skk_{geometry}'
    print(f'--- building {tag} ({"3" if stacked else "2"} domains, {pol}-pol)', flush=True)
    model = client.create(tag); jm = model.java

    # ------------------------------------------------------------ parameters
    params = [('lam', '800[nm]', 'wavelength'),
              ('f0', 'c_const/lam', ''),
              ('theta', '0[deg]', 'incidence angle, in the background medium'),
              ('n_bg', '1.7', 'background index'),
              ('Lx', 'lam/4', 'period: only diffraction order 0 propagates'),
              ('L_bg', 'lam', 'background thickness'),
              ('L_term', '8*lam', 'TERMINATOR thickness (PML or sKK -- same domain)'),
              ('N_term', '400', 'elements across the terminator'),
              ('N_bg', '40', 'elements across the background'),
              ('N_x', '5', 'elements across the period')]
    if stacked:
        params += [('L_low', '2*lam', 'lower region thickness'),
                   ('N_low', '400', 'elements across the lower region'),
                   ('eps_low', 'n_bg^2', 'lower region permittivity')]
    for k, v, d in params:
        jm.param().set(k, v, d)

    comp = jm.component().create('comp1', True)
    geom = comp.geom().create('geom1', 2)
    r1 = geom.create('r1', 'Rectangle')
    height = 'L_low+L_term+L_bg' if stacked else 'L_term+L_bg'
    r1.set('size', jpype.JArray(jpype.JString)(['Lx', height]))
    names = ['low', 'term'] if stacked else ['term']
    r1.set('layername', jpype.JArray(jpype.JString)(names))
    for i, ln in enumerate(['L_low', 'L_term'] if stacked else ['L_term']):
        r1.setIndex('layer', ln, i)
    geom.run()

    # ------------------------------------------------------------ topology
    nb_ = int(geom.getNBoundaries()); vc = np.array(geom.getVertexCoord())
    adj = [list(np.array(r).ravel()) for r in geom.getAdj(1, 0)]
    mid = {b: vc[:, [int(v)-1 for v in adj[b]]].mean(axis=1) for b in range(1, nb_+1)}
    ud = np.array(geom.getUpDown())
    xs = sorted({round(m[0], 12) for m in mid.values()}); xlo, xhi = xs[0], xs[-1]
    verts = [b for b, m in mid.items() if abs(m[0]-xlo) < 1e-12 or abs(m[0]-xhi) < 1e-12]
    horiz = [b for b in mid if b not in verts]
    b_top = max(horiz, key=lambda b: mid[b][1]); b_bot = min(horiz, key=lambda b: mid[b][1])
    dy = {}
    for b in verts:
        for d in (int(ud[0][b-1]), int(ud[1][b-1])):
            if d > 0:
                dy.setdefault(d, []).append(mid[b][1])
    doms = sorted(dy, key=lambda d: np.mean(dy[d]))          # bottom -> top
    if stacked:
        d_low, d_term, d_bg = doms
    else:
        d_term, d_bg = doms; d_low = None
    print(f'    domains term={d_term} bg={d_bg}' + (f' low={d_low}' if stacked else '')
          + f'; port={b_top} pec={b_bot}', flush=True)

    # ------------------------------------------------------------ the profile function
    os.makedirs(S.PROFILES, exist_ok=True)
    tmpl = S.write_profile('_template_flat',
                           S.profile('flat', 8*S.LAM, eps_b=S.EB_DEFAULT, npts=2))
    # ORDER MATTERS: filename first, THEN struct. Set the other way round, `struct` reverts
    # to 'grid' and a 3-column file is read as a 2-D grid -- the function then wants two
    # arguments and every solve dies with "Function called with wrong number of arguments".
    # ('nargs' is a table-source property and is rejected outright for a file source.)
    def mkfun(name, path):
        """One interpolation function per part, each reading a TWO-column file.

        Three gotchas, all of which fail only at solve time with messages that never
        mention the interpolation file:
          * The function is named after the FEATURE TAG. `funcname` is ignored for a file
            source, so the tags have to be 'p_re'/'p_im'.
          * A single 3-column file does not work: the spreadsheet format reads the leftmost
            columns as arguments, so `y eps_re eps_im` is taken as a function of TWO
            arguments -> "Function called with wrong number of arguments".
          * Set `filename` BEFORE `struct`, or struct silently reverts to 'grid'.
        """
        f = comp.func().create(name, 'Interpolation')
        f.set('source', 'file')
        f.set('filename', str(path))
        f.set('struct', 'spreadsheet')
        f.set('interp', 'linear'); f.set('extrap', 'const')
        f.set('argunit', 'm'); f.set('fununit', '1')
        f.importData()
        assert str(f.getString('struct')) == 'spreadsheet', 'struct did not stick'
        assert str(f.getStringMatrix('funcs')[0][0]) == name, 'function name is not the tag'
        f.label(f'terminator {name} (from profiles/*.{name[-2:]}.txt)')
        return f

    mkfun('p_re', tmpl + '.re.txt')
    mkfun('p_im', tmpl + '.im.txt')

    # ------------------------------------------------------------ physics
    m1 = comp.material().create('mat1', 'Common'); m1.label('background')
    m1.propertyGroup('def').set('relpermeability',
        jpype.JArray(jpype.JString)(['1', '0', '0', '0', '1', '0', '0', '0', '1']))
    m1.propertyGroup('def').set('electricconductivity', jpype.JArray(jpype.JString)(['0']*9))

    pml = comp.coordSystem().create('pml1', 'PML')
    pml.selection().set(jpype.JArray(jpype.JInt)([d_low if stacked else d_term]))
    pml.set('PMLfactor', '1'); pml.active(False)
    pml.label('PML (inactive by default; occupies the '
              + ('lower region' if stacked else 'TERMINATOR domain') + ')')

    ph = comp.physics().create('ewfd', 'ElectromagneticWavesFrequencyDomain', 'geom1')
    ph.prop('EquationForm').set('form', 'Frequency')
    ph.prop('EquationForm').set('freq_src', 'userdef')
    ph.prop('EquationForm').set('freq', 'f0')
    ph.prop('components').set('components',
                              'outofplane' if pol == 's' else 'inplane')
    w1 = ph.feature('wee1'); w1.set('DisplacementFieldModel', 'DielectricLoss')
    w1.set('minput_frequency_src', 'userdef'); w1.set('minput_frequency', 'f0')
    w1.set('epsilonPrim_mat', 'userdef'); w1.set('epsilonPrim', 'n_bg^2')
    w1.set('epsilonBis_mat', 'userdef');  w1.set('epsilonBis', '0')

    def wave_eq(tag, dom, label, eps_re, eps_im):
        w = ph.create(tag, 'WaveEquationElectric', 2)
        w.selection().set(jpype.JArray(jpype.JInt)([dom])); w.label(label)
        w.set('DisplacementFieldModel', 'DielectricLoss')
        w.set('epsilonPrim_mat', 'userdef'); w.set('epsilonPrim', eps_re)
        w.set('epsilonBis_mat', 'userdef');  w.set('epsilonBis', eps_im)
        w.set('mur_mat', 'userdef'); w.set('mur', '1')
        w.set('sigma_mat', 'userdef'); w.set('sigma', '0')
        return w

    arg = 'y-L_low' if stacked else 'y'
    wave_eq('wee_term', d_term, 'terminator', f'p_re({arg})', f'p_im({arg})')
    if stacked:
        wave_eq('wee_low', d_low, 'lower region', 'eps_low', '0')

    pc = ph.create('pc1', 'PeriodicCondition', 1)
    pc.selection().set(jpype.JArray(jpype.JInt)(verts))
    pc.set('PeriodicType', 'Floquet'); pc.set('Floquet_source', 'FromPeriodicPort')
    pt = ph.create('port1', 'Port', 1)
    pt.selection().set(jpype.JArray(jpype.JInt)([b_top]))
    pt.set('PortType', 'Periodic'); pt.set('alpha_inc', 'theta')
    E0 = ([['0'], ['0'], ['1']] if pol == 's'          # s-pol: E out of plane (Ez)
          else [['cos(theta)'], ['sin(theta)'], ['0']])  # p-pol: E in the plane of incidence
    pt.set('E0', jpype.JArray(jpype.JString, 2)(E0))
    pt.set('Eampl', jpype.JArray(jpype.JString, 2)(E0))

    # ------------------------------------------------------------ mesh
    msh = comp.mesh().create('mesh1', 'geom1')
    left = sorted([b for b in verts if abs(mid[b][0]-xlo) < 1e-12], key=lambda b: mid[b][1])
    edges = ([('e_low', 'N_low'), ('e_term', 'N_term'), ('e_bg', 'N_bg')] if stacked
             else [('e_term', 'N_term'), ('e_bg', 'N_bg')])
    for i, (etag, n) in enumerate(edges):
        e = msh.create(etag, 'Edge'); e.selection().set(jpype.JArray(jpype.JInt)([left[i]]))
        e.create('dis1', 'Distribution').set('numelem', n)
    e = msh.create('e_x', 'Edge'); e.selection().set(jpype.JArray(jpype.JInt)([b_bot]))
    e.create('dis1', 'Distribution').set('numelem', 'N_x')
    for i in range(len(edges)):
        c = msh.create(f'cp{i+1}', 'CopyEdge')
        c.selection('source').set(jpype.JArray(jpype.JInt)([left[i]]))
        rt = [b for b in verts if abs(mid[b][0]-xhi) < 1e-12
              and abs(mid[b][1]-mid[left[i]][1]) < 1e-12]
        c.selection('destination').set(jpype.JArray(jpype.JInt)(rt))
    msh.create('map1', 'Map')
    msh.run()

    # ------------------------------------------------------------ study
    std = jm.study().create('std1'); std.create('stat', 'Stationary')
    st = std.feature('stat')
    st.set('useparam', True)
    st.set('pname', jpype.JArray(jpype.JString)(['theta']))
    st.set('plistarr', jpype.JArray(jpype.JString)(['0 45 80 89']))
    st.set('punit', jpype.JArray(jpype.JString)(['deg']))
    jm.sol().create('sol1').attach('std1'); jm.sol('sol1').createAutoSequence('std1')

    if save:
        os.makedirs(MODELS, exist_ok=True)
        out = os.path.join(MODELS, f'{tag}.mph')
        model.save(out)
        print(f'    saved {out}  ({os.path.getsize(out)/1e6:.2f} MB)', flush=True)
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--geometry', choices=['swap', 'stacked', 'both'], default='both')
    ap.add_argument('--pol', choices=['s', 'p'], default='s')
    a = ap.parse_args()
    t0 = time.time(); client = mph.start()
    for g in (['swap', 'stacked'] if a.geometry == 'both' else [a.geometry]):
        build(client, g, a.pol)
    client.clear()
    print(f'done in {time.time()-t0:.0f}s -- NO solves were run', flush=True)
