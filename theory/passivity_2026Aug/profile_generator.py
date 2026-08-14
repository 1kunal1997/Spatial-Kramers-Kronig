"""Passivity profile generator — on-demand (eps', eps'') pairs with labels.

Every profile is defined by its REAL part eps'(x) on a periodic domain [-L, L).
The imaginary part is produced by DiHiTI (tmm_helper.ht_derivative), and each
sample is labelled with:

    min_eim  : min eps''(x)          < 0  =>  the profile has GAIN
    lam_min  : smallest eigenvalue of the Toeplitz matrix built from eps'''s
               Fourier coefficients with c_0 stripped.  Carathéodory-Toeplitz
               says the profile is passive iff lam_min + mean(eps'') >= 0, and
               C_g_min = -lam_min is the minimum uniform background loss that
               would make it passive.

Use as a library:
    from profile_generator import make, PROFILES, sample
    s = make('logistic', k=4.0)          # -> dict(x, e_re, e_im, min_eim, ...)
    for s in sample(200): ...            # random draws across all families

Use from the shell:
    python profile_generator.py --list
    python profile_generator.py logistic --k 4.0
    python profile_generator.py --gallery          # every family, one figure
"""
import sys, os, argparse
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'theory'))

import numpy as np
from scipy.linalg import toeplitz as _toeplitz
import tmm_helper as tmm_h

NB, EPS_B, EPS_AIR = 1.7, 1.7**2, 1.0
L_DEF, N_DEF = 2.5, 2048
A_BUMP = EPS_B - EPS_AIR


def grid(L=L_DEF, N=N_DEF):
    return np.linspace(-L, L, N)


def _g(x, x0, s):
    return np.exp(-((x - x0) / s)**2)


def _inv_ht(x, e_im, anchor=EPS_AIR):
    """Backwards construction: prescribe eps'' >= 0, recover eps' = -H[eps''] + c."""
    r = -tmm_h.np.imag(tmm_h.hilbert(e_im)) if hasattr(tmm_h, 'hilbert') else None
    if r is None:
        from scipy.signal import hilbert as _h
        r = -np.imag(_h(e_im))
    return r + (anchor - r[-1])


# --------------------------------------------------------------- families
# Each entry: name -> (callable(x, **kw) -> eps'(x), default kwargs)
def _logistic(x, k=4.0, nb=NB):
    return tmm_h.logistic(x, k, nb)


def _lorentzian(x, A=1.0, gam=0.5, nb=NB):
    return np.real(tmm_h.eps(x, A, gam, nb))


def _logistic_bump(x, k=4.0, alpha=1.0, x0=0.0, w=0.6):
    return tmm_h.logistic(x, k, NB) + alpha * A_BUMP * _g(x, x0, w)


def _opposite_bumps(x, k=4.0, alpha=1.0, sep=0.9, w=0.45):
    return (tmm_h.logistic(x, k, NB)
            + alpha * A_BUMP * (_g(x, -sep, w) - _g(x, sep, w)))


def _triangle(x, slope=2.0, tip=0.15):
    L = x[-1]
    return EPS_AIR + slope * (np.sqrt(L**2 + tip**2) - np.sqrt(x**2 + tip**2))


def _cosine_bump(x, beta=0.0):
    L = x[-1]
    return EPS_AIR + (A_BUMP / 2) * (1 + np.cos(np.pi * x / L)) * (1 + beta * x)


def _sine(x, amp=0.945, sign=+1.0):
    L = x[-1]
    return EPS_B + sign * amp * np.sin(np.pi * x / L) - amp


def _gaussian_re(x, sL=2.0, sR=2.0):
    L = x[-1]
    sig = sL * (L - x) / (2 * L) + sR * (x + L) / (2 * L)
    return EPS_AIR + A_BUMP * np.exp(-x**2 / (2 * sig**2))


def _tanh_slopes(x, mL=2.0, mR=1.0, kappa=4.0):
    L = x[-1]
    b, d = (mL + mR) / 2, (mR - mL) / 2
    a = (EPS_B + mL * L + EPS_AIR - mR * L) / 2
    c = (EPS_AIR - mR * L - EPS_B - mL * L) / 2
    return a + b * x + c * np.tanh(kappa * x) + d * x * np.tanh(kappa * x)


# --- backwards families: prescribe a NON-NEGATIVE eps'' and invert (always passive)
def _bwd_mesa(x, amp=1.2, edge=1.2, k=4.0):
    return _inv_ht(x, amp * (np.tanh(k * (x + edge)) - np.tanh(k * (x - edge))) / 2)


def _bwd_gauss(x, amp=1.5, w=0.8):
    return _inv_ht(x, amp * _g(x, 0.0, w))


def _bwd_double(x, a1=0.8, x1=-0.9, a2=1.4, x2=0.5, w=0.5):
    return _inv_ht(x, a1 * _g(x, x1, w) + a2 * _g(x, x2, w))


PROFILES = {
    'logistic':       (_logistic,       dict(k=4.0)),
    'lorentzian':     (_lorentzian,     dict(A=1.0, gam=0.5)),
    'logistic_bump':  (_logistic_bump,  dict(alpha=1.0, x0=0.0, w=0.6)),
    'opposite_bumps': (_opposite_bumps, dict(alpha=1.0, sep=0.9, w=0.45)),
    'triangle':       (_triangle,       dict(slope=2.0)),
    'cosine_bump':    (_cosine_bump,    dict(beta=0.0)),
    'sine':           (_sine,           dict(amp=0.945, sign=+1.0)),
    'gaussian':       (_gaussian_re,    dict(sL=2.0, sR=2.0)),
    'tanh_slopes':    (_tanh_slopes,    dict(mL=2.0, mR=1.0)),
    'bwd_mesa':       (_bwd_mesa,       dict(amp=1.2, edge=1.2)),
    'bwd_gauss':      (_bwd_gauss,      dict(amp=1.5, w=0.8)),
    'bwd_double':     (_bwd_double,     dict(a1=0.8, a2=1.4)),
}

# ranges used by sample() for random draws
RANGES = {
    'logistic':       dict(k=(1.0, 12.0)),
    'lorentzian':     dict(A=(0.2, 6.0), gam=(0.1, 1.5)),
    'logistic_bump':  dict(alpha=(-2.0, 3.0), x0=(-1.5, 1.5), w=(0.25, 1.2)),
    'opposite_bumps': dict(alpha=(0.0, 2.5), sep=(0.4, 1.8), w=(0.25, 0.8)),
    'triangle':       dict(slope=(0.2, 5.0)),
    'cosine_bump':    dict(beta=(-0.38, 0.38)),
    'sine':           dict(amp=(0.2, 1.5), sign=(-1.0, 1.0)),
    'gaussian':       dict(sL=(0.5, 3.0), sR=(0.5, 3.0)),
    'tanh_slopes':    dict(mL=(-4.0, 6.0), mR=(-4.0, 6.0)),
    'bwd_mesa':       dict(amp=(0.4, 2.5), edge=(0.4, 2.0)),
    'bwd_gauss':      dict(amp=(0.3, 3.0), w=(0.3, 1.6)),
    'bwd_double':     dict(a1=(0.1, 2.0), a2=(0.1, 2.0)),
}


# ------------------------------------------------------------- diagnostics
def toeplitz_lam_min(e_im, M=128):
    """Smallest eigenvalue of the Toeplitz form with the gauge constant stripped.
    C_g_min = -lam_min is the minimum background loss for passivity."""
    n = len(e_im)
    c = np.fft.fft(e_im)[:M + 1] / n
    c = c.copy()
    c[0] = 0.0
    return float(np.linalg.eigvalsh(_toeplitz(c, np.conj(c))).min())


def make(name, L=L_DEF, N=N_DEF, **params):
    """Build one labelled sample."""
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    fn, defaults = PROFILES[name]
    kw = dict(defaults)
    kw.update(params)
    x = grid(L, N)
    e_re = np.asarray(fn(x, **kw), float)
    e_im = tmm_h.ht_derivative(x, e_re)
    lam = toeplitz_lam_min(e_im)
    fom, _, _ = tmm_h.skk_spectral_fom(x, e_re, e_im)
    # Tolerance must be RELATIVE to the profile's own loss scale: an absolute
    # cutoff mislabels discretisation noise (~1e-5) as gain on weak profiles,
    # which would poison any downstream pattern search.
    scale = max(float(e_im.max()), 1e-12)
    return dict(name=name, params=kw, x=x, e_re=e_re, e_im=e_im,
                min_eim=float(e_im.min()), lam_min=lam, Cg_min=max(0.0, -lam),
                Cg_actual=float(e_im.mean()), fom=float(fom),
                gain_rel=float(-e_im.min() / scale),
                passive=bool(e_im.min() >= -1e-3 * scale))


def sample(n, rng=None, names=None):
    """Yield n random labelled samples across the families."""
    rng = rng or np.random.default_rng(0)
    names = names or sorted(PROFILES)
    for _ in range(n):
        nm = names[rng.integers(len(names))]
        kw = {}
        for p, (a, b) in RANGES.get(nm, {}).items():
            v = rng.uniform(a, b)
            if p == 'sign':
                v = 1.0 if v > 0 else -1.0
            kw[p] = v
        yield make(nm, **kw)


# ------------------------------------------------------------------ plotting
def plot(specs, outfile, ncols=3):
    """Dual-axis eps'/eps'' panels using the project's skk_fig_common presets."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import skk_fig_common as sfc

    if isinstance(specs, dict):
        specs = [specs]
    sfc.apply_style()
    ncols = min(ncols, len(specs))
    nrows = int(np.ceil(len(specs) / ncols))
    fig, axs, _ = sfc.panel_grid(nrows, ncols, has_twin=True)
    axs = np.atleast_2d(axs)                      # relayout_grid needs the 2-D array
    flat = axs.ravel()
    for ax, s in zip(flat, specs):
        sfc.profile_panel(ax, s['x'], s['e_re'], s['e_im'], xlabel=r'$x$ ($\mu$m)',
                          xlim=(s['x'][0], s['x'][-1]))
        tag = 'PASSIVE' if s['passive'] else f"GAIN {s['min_eim']:.3f}"
        ax.set_title(f"{s['name']}  [{tag}]", fontsize=7)
    for ax in flat[len(specs):]:
        ax.set_visible(False)
    sfc.relayout_grid(fig, axs)
    fig.savefig(outfile, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {outfile}')


# ---------------------------------------------------------------------- CLI
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('name', nargs='?', help='profile family name')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--gallery', action='store_true', help='one panel per family')
    ap.add_argument('--outdir', default=os.path.dirname(os.path.abspath(__file__)))
    args, extra = ap.parse_known_args()

    if args.list:
        print(f"{'family':<18}{'defaults':<44}{'min eps_im':>12}{'lam_min':>11}{'verdict':>10}")
        for nm in sorted(PROFILES):
            s = make(nm)
            print(f"{nm:<18}{str(s['params']):<44}{s['min_eim']:>12.5f}"
                  f"{s['lam_min']:>11.5f}{'passive' if s['passive'] else 'GAIN':>10}")
        sys.exit(0)

    if args.gallery:
        specs = [make(nm) for nm in sorted(PROFILES)]
        plot(specs, os.path.join(args.outdir, 'profile_gallery.png'))
        sys.exit(0)

    if not args.name:
        ap.error('give a family name, or --list / --gallery')

    kw = {}
    for i in range(0, len(extra), 2):
        kw[extra[i].lstrip('-')] = float(extra[i + 1])
    s = make(args.name, **kw)
    print(f"{s['name']}  {s['params']}")
    print(f"  min eps''      = {s['min_eim']:+.6f}   -> {'PASSIVE' if s['passive'] else 'GAIN'}")
    print(f"  lam_min(T)     = {s['lam_min']:+.6f}")
    print(f"  C_g,min        = {s['Cg_min']:.6f}   (DiHiTI gauge = {s['Cg_actual']:.6f})")
    print(f"  spectral FoM   = {s['fom']:.6f} %")
    plot(s, os.path.join(args.outdir, f"profile_{s['name']}.png"))
