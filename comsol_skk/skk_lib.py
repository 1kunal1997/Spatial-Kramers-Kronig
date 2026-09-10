"""Shared harness for the sKK-as-PML study: profiles, models, meshes, solves, data files.

Every experiment script (`exp_*.py`) imports this. Nothing else defines a profile, touches
the COMSOL API, or writes a data file, so a figure can always be traced to one definition.

GEOMETRY CONVENTION (both models)
    y = 0            the PEC-backed end of the terminator
    y = L_term       the ENTRY face, where the wave arrives from the background
    the wave travels in -y.  Angles are measured in the background medium.

PROFILE CONVENTION
    profile(kind, L, eps_b, **params) -> Profile(y, eps_re, eps_im, meta)
    with y in [0, L] in metres and eps_re/eps_im the relative permittivity.
    eps_im >= 0 is loss (COMSOL 'DielectricLoss' with a positive epsilonBis).
    Feature widths are REQUIRED arguments -- there is deliberately no default, because
    two different width rules (w = L/40, w = L/80, w = lam/5) were used in the Aug-Sep
    2026 runs and the only way to tell them apart was to read the script.

DATA CONVENTION
    save_result(path, results, meta) writes <name>.npz holding every result array plus a
    'meta' entry: a JSON blob of every parameter that produced it, the profile file used,
    and the git commit. A figure must be answerable from the data file alone.
"""
import os, sys, json, subprocess, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import tmm_helper as tmm_h

MODELS   = os.path.join(HERE, 'Models')
PROFILES = os.path.join(HERE, 'profiles')
DATA     = os.path.join(HERE, 'data')

LAM = 0.8e-6                     # the study's wavelength, 800 nm
NB_DEFAULT = 1.7
EB_DEFAULT = NB_DEFAULT**2


# ==================================================================== profiles
def _logistic(y, L, eps_b, w):
    """eps' steps eps_b (entry) -> 1 (PEC end); eps'' is its Hilbert partner (DiHiTI).
    Terminates in air, so it needs a background of index > 1 to carry any loss."""
    er = 1.0 + (eps_b - 1.0)/(1.0 + np.exp(-(y - L/2)/w))
    return er, -tmm_h.ht_derivative(y, er)


def _turned_logistic(y, L, eps_b, w, amp=1.0):
    """The logistic pair turned by 90 deg: eps'' carries the step (0 at the entry face,
    `amp` at the PEC end) and eps' is its Hilbert partner, shifted to eps_b at both ends.
    Same one-sided analytic function multiplied by e^{-i pi/2}, so the sKK property is
    exact. eps' >= eps_b everywhere -> no critical angle at any incidence."""
    ei = amp/(1.0 + np.exp((y - L/2)/w))
    return eps_b + tmm_h.ht_derivative(y, ei), ei


def _lorentzian(y, L, eps_b, gam, amp=4.0):
    """eps_b on BOTH sides -- no index step, no TIR, honest at every angle.
    Exact closed form: eps = eps_b + i*A*gam/((y-y0) + i*gam)."""
    u = y - L/2; D = u**2 + gam**2
    return eps_b + amp*gam*u/D, amp*gam**2/D


def _tapered(y, L, eps_b, gam, ell, amp=4.0):
    """Lorentzian with a sech taper on eps' only (eps'' left alone, so still passive).
    NOT an exact Hilbert pair -- the taper is what trades KK mismatch for a smaller
    truncation step."""
    u = y - L/2; D = u**2 + gam**2
    return eps_b + amp*gam*u/D/np.cosh(u/ell), amp*gam**2/D


def _flat(y, L, eps_b):
    """Uniform background. Used for the terminator domain in PML runs, so the PML and the
    sKK profiles occupy the SAME domain and share its thickness parameter."""
    return np.full_like(y, eps_b), np.zeros_like(y)


_KINDS = {'logistic': _logistic, 'turned_logistic': _turned_logistic,
          'lorentzian': _lorentzian, 'tapered': _tapered, 'flat': _flat}

#: which value eps' takes at the PEC end -- what a stacked model's lower region must match
TERMINATES_IN = {'logistic': 'air', 'turned_logistic': 'eps_b',
                 'lorentzian': 'eps_b', 'tapered': 'eps_b', 'flat': 'eps_b'}


class Profile:
    __slots__ = ('y', 'eps_re', 'eps_im', 'meta')

    def __init__(self, y, eps_re, eps_im, meta):
        self.y, self.eps_re, self.eps_im, self.meta = y, eps_re, eps_im, meta

    @property
    def eps_entry(self):
        return complex(self.eps_re[-1], self.eps_im[-1])

    @property
    def eps_pec(self):
        return complex(self.eps_re[0], self.eps_im[0])

    def passive(self, tol=1e-9):
        return bool(self.eps_im.min() > -tol)

    def one_sidedness(self):
        """Spectral one-sidedness in percent; 100 = perfectly one-sided.
        Checks both handedness conventions and returns the larger."""
        if np.ptp(self.eps_re) == 0 and np.ptp(self.eps_im) == 0:
            return float('nan')            # a uniform medium has no spectrum to be one-sided
        a = tmm_h.skk_spectral_fom(self.y, self.eps_re, self.eps_im, allowed_side='positive')[0]
        b = tmm_h.skk_spectral_fom(self.y, self.eps_re, self.eps_im, allowed_side='negative')[0]
        return max(a, b)


def n_points(L, feature):
    """Sampling rule for the profile table: spacing <= min(feature/40, 0.002 lam), i.e. at
    least 10x finer than the finest mesh we would ever use. Kept well clear of the mesh so
    the table is never the limiting discretization."""
    dy = min(feature/40.0, 0.002*LAM)
    return int(np.clip(round(L/dy) + 1, 2001, 400001))


def profile(kind, L, eps_b=EB_DEFAULT, npts=None, **params):
    """Build a profile on y in [0, L]. Feature width must be given explicitly."""
    if kind not in _KINDS:
        raise ValueError(f'unknown profile {kind!r}; have {sorted(_KINDS)}')
    feature = params.get('w') or params.get('gam') or L/40.0
    npts = npts or (n_points(L, feature) if kind != 'flat' else 2)
    y = np.linspace(0.0, L, npts)
    er, ei = _KINDS[kind](y, L, eps_b, **params)
    meta = dict(kind=kind, L_over_lam=L/LAM, eps_b=eps_b, npts=int(npts),
                terminates_in=TERMINATES_IN[kind],
                **{k: (v/LAM if k in ('w', 'gam', 'ell') else v) for k, v in params.items()})
    if any(k in params for k in ('w', 'gam', 'ell')):
        meta['width_units'] = 'lambda'
    return Profile(y, er, ei, meta)


def write_profile(name, prof, outdir=PROFILES):
    """Write the profile as TWO two-column COMSOL spreadsheet files:

        profiles/<name>.re.txt      % header, then `y eps_re`
        profiles/<name>.im.txt      % header, then `y eps_im`

    Two columns rather than one three-column file because the spreadsheet format reads the
    leftmost columns as arguments: `y eps_re eps_im` is taken as a function of TWO arguments
    and every solve fails. Returns the path PREFIX (no extension); pass it to set_profile.

    The same files are what a figure script reads back, so a figure cannot disagree with
    what COMSOL was fed.
    """
    os.makedirs(outdir, exist_ok=True)
    prefix = os.path.join(outdir, name)
    head = (f'% sKK terminator profile: {name}\n'
            f'% written {datetime.date.today().isoformat()} by skk_lib.write_profile\n'
            f'% {json.dumps(prof.meta)}\n'
            f'% passive={prof.passive()}  one_sidedness={prof.one_sidedness():.4f}%\n')
    for part, col in (('re', prof.eps_re), ('im', prof.eps_im)):
        with open(f'{prefix}.{part}.txt', 'w') as fh:
            fh.write(head + f'% y[m] eps_{part}\n')
            np.savetxt(fh, np.column_stack([prof.y, col]), fmt='%.10e')
    return prefix


def read_profile(prefix):
    """Read a profile back from its two files (for figures -- no COMSOL needed).
    `prefix` is what write_profile returned. Returns (y, eps_re, eps_im, meta_dict)."""
    meta = {}
    with open(f'{prefix}.re.txt') as fh:
        for line in fh:
            if not line.startswith('%'):
                break
            body = line[1:].strip()
            if body.startswith('{'):
                meta = json.loads(body)
    re_ = np.loadtxt(f'{prefix}.re.txt', comments='%')
    im_ = np.loadtxt(f'{prefix}.im.txt', comments='%')
    return re_[:, 0], re_[:, 1], im_[:, 1], meta


# ==================================================================== model
def load(client, geometry='swap'):
    """Open one of the two template models. Never saved back -- experiments change
    parameters in memory only, so the .mph on disk stays the reviewable template."""
    path = os.path.join(MODELS, f'skk_{geometry}.mph')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} -- build it with: python model_build.py')
    return client.load(path)


def set_profile(model, prefix):
    """Point the terminator's two interpolation functions at a profile and re-read it.
    `prefix` is what write_profile returned (no extension).

    Order matters twice: discardData() before the new filename (otherwise importData
    throws), and `struct` after `filename` (set the other way round it reverts to 'grid'
    and the solve fails with a message that never mentions the file).
    """
    comp = model.java.component('comp1')
    for tag, part in (('p_re', 're'), ('p_im', 'im')):
        f = comp.func(tag)
        f.discardData()                       # must come BEFORE the new filename
        f.set('filename', f'{prefix}.{part}.txt')
        f.set('struct', 'spreadsheet')
        f.importData()
        if str(f.getString('struct')) != 'spreadsheet':
            raise RuntimeError(f'{tag}: struct is {f.getString("struct")!r}, not spreadsheet')


def set_mesh(model, n_term, grading='uniform', ratio=1.0):
    """Element count and distribution across the terminator.

    grading: 'uniform'      equidistant (the default everywhere)
             'centre'       predefined+geometric, symmetric+reverse -> fine in the middle
             'entry'        predefined+geometric, reverse           -> fine at the entry face
             'pec'          predefined+geometric                    -> fine at the PEC end
    Measured on this harness: uniform beats every graded option for these profiles, because
    the binding constraint is resolving the WAVE, not the eps feature. Kept because the
    question comes up.
    """
    model.parameter('N_term', str(int(n_term)))
    dis = model.java.component('comp1').mesh('mesh1').feature('e_term').feature('dis1')
    for k in ('symmetric', 'reverse'):
        dis.set(k, 'off')
    if grading == 'uniform':
        dis.set('type', 'number'); dis.set('numelem', 'N_term'); return
    dis.set('type', 'predefined'); dis.set('method', 'geometric')
    dis.set('elemcount', 'N_term'); dis.set('elemratio', str(ratio))
    if grading in ('entry', 'centre'):
        dis.set('reverse', 'on')
    if grading == 'centre':
        dis.set('symmetric', 'on')


def set_pml(model, active, pmlfactor=1):
    pml = model.java.component('comp1').coordSystem('pml1')
    pml.active(bool(active))
    if active:
        pml.set('PMLfactor', str(pmlfactor))


def solve_angles(model, angles):
    """Parametric solve over incidence angle. Returns (theta_deg, R) as float arrays."""
    import jpype
    ang = np.atleast_1d(np.asarray(angles, float))
    plist = ' '.join(f'{a:g}' for a in ang)
    st = model.java.study('std1').feature('stat')
    st.set('plistarr', jpype.JArray(jpype.JString)([plist]))
    model.solve(model.studies()[0])
    R = np.asarray(model.evaluate('ewfd.Rtotal')).ravel()
    return ang[:len(R)], R


# ==================================================================== data
def git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def save_result(name, results, meta, outdir=DATA):
    """Write data/<name>.npz: every result array plus a 'meta' JSON blob."""
    os.makedirs(outdir, exist_ok=True)
    meta = dict(meta)
    meta.update(experiment=name, written=datetime.datetime.now().isoformat(timespec='seconds'),
                git=git_hash(), lam_m=LAM)
    path = os.path.join(outdir, f'{name}.npz')
    np.savez(path, meta=np.array(json.dumps(meta, default=str)), **results)
    print(f'  -> {path}  ({len(results)} arrays)')
    return path


def load_result(name, outdir=DATA):
    """Returns (dict_of_arrays, meta_dict)."""
    d = np.load(os.path.join(outdir, f'{name}.npz'), allow_pickle=True)
    meta = json.loads(str(d['meta']))
    return {k: d[k] for k in d.files if k != 'meta'}, meta
