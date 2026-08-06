"""SI Fig 1 — raw reflectance / absorptance landscapes behind the main-text enhancement maps.

Supersedes _mockup_fig3_SI.py (3-panel pyramid, s-pol only, colorbars outside the axes).
Layout is 3 rows x 2 columns:
  rows = quantity   (R_sKK, R_GRIN, A_sKK)
  cols = polarization (s left, p right)
Each row carries its OWN vertical colorbar at the right edge. Rows 0 and 1 deliberately share
IDENTICAL colour limits -- the second bar is redundant in information but keeps the two heatmap
rows the same width, so the block reads as one organised grid instead of two ragged rows.

Grid is EXACTLY the main-text ratio-map grid of _mockup_fig3_4_combined.py (theta_b step 0.5 deg,
lambda/L in [0.06, 30] at 30 points per decade, delta = DELTA_COLORPLOT), so these raw maps overlay
the enhancement maps cell for cell. The four R maps are therefore reused straight from that
figure's cache when its metadata matches; only the two absorptance maps are new.

Note on the colour floor: the old SI figure pinned vmin at 1e-8, which SATURATED the whole
lambda < L/5 corner of the R_sKK map. That was a colourbar clip, not a numerical floor -- the
true minimum of these maps is ~6e-12, far above double-precision underflow. Here vmin is taken
from the data (rounded to a decade) so the structure is visible and nothing is hidden.
"""
import sys, os, json
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import (FixedLocator, FixedFormatter, LogLocator,
                               NullFormatter)
import theory.skk_fig_common as C

C.apply_style()

# ---------------------------------------------------------------------------
# Grid — must match _mockup_fig3_4_combined.py EXACTLY (that is the whole point)
# ---------------------------------------------------------------------------
LOL_MIN, LOL_MAX = 6.0 / 100.0, 30.0
PTS_PER_DECADE = 30
ANG_STEP = 0.5
LOL_TICKS = [0.1, 0.2, 1, 2, 10]
LOL_LABELS = [r'$L/10$', r'$L/5$', r'$L$', r'$2L$', r'$10L$']

RATIO_CACHE = os.path.join(_PROJECT_ROOT, 'theory', 'Data', 'fig3_4_ratio')
ABS_CACHE = os.path.join(_PROJECT_ROOT, 'theory', 'Data', 'SI_colorplots')

xx, e_re, e_im = C.logistic_profile()
build_skk = lambda: C.build_stack(xx, e_re + 1j * e_im, n_out=1.0, delta=C.DELTA_COLORPLOT)
build_grin = lambda: C.build_stack(xx, e_re + 0j, n_out=1.0, delta=C.DELTA_COLORPLOT)
# thickness comes from the PRIMARY-delta stack, as every other figure reports it
L_tot = C.coating_thickness(C.build_stack(xx, e_re + 1j * e_im, n_out=1.0)[1])
print(f"coating thickness L = {L_tot:.3f} um")

ang = C.theta_b_deg(step=ANG_STEP)
n_lol = int(round(np.log10(LOL_MAX / LOL_MIN) * PTS_PER_DECADE)) + 1
lol = np.logspace(np.log10(LOL_MIN), np.log10(LOL_MAX), n_lol)
lams = lol * L_tot
print(f"grid: {len(ang)} angles x {len(lol)} wavelengths")

meta = dict(delta=C.DELTA_COLORPLOT, nb=C.NB, L=round(L_tot, 6),
            ang_step=ANG_STEP, lol_min=LOL_MIN, lol_max=LOL_MAX,
            pts_per_decade=PTS_PER_DECADE)


def _cached(cache_dir, keys, compute):
    """Load {key: array} from cache_dir when its metadata.json matches `meta`, else compute."""
    os.makedirs(cache_dir, exist_ok=True)
    mpath = os.path.join(cache_dir, 'metadata.json')
    files = {k: os.path.join(cache_dir, f'{k}.npy') for k in keys}
    if os.path.exists(mpath) and all(os.path.exists(p) for p in files.values()):
        with open(mpath) as f:
            if json.load(f) == meta:
                print(f"  [cache hit] {os.path.basename(cache_dir)}: {', '.join(keys)}")
                return {k: np.load(p) for k, p in files.items()}
    print(f"  [cache miss] computing {', '.join(keys)} -> {cache_dir}")
    out = compute()
    for k, p in files.items():
        np.save(p, out[k])
    with open(mpath, 'w') as f:
        json.dump(meta, f, indent=2)
    return out


def _compute_R():
    n_g, d_g = build_grin()
    n_s, d_s = build_skk()
    out = {}
    for pol in ('s', 'p'):
        out[f'grin_{pol}'], _ = C.tra_map(n_g, d_g, ang, lams, pol=pol)
        out[f'skk_{pol}'], _ = C.tra_map(n_s, d_s, ang, lams, pol=pol)
    return out


def _compute_A():
    n_s, d_s = build_skk()
    out = {}
    for pol in ('s', 'p'):
        print(f"    absorptance map, {pol}-pol ({len(ang)}x{len(lol)} TMM solves)...")
        _, out[f'askk_{pol}'] = C.tra_map(n_s, d_s, ang, lams, pol=pol)
    return out


R = _cached(RATIO_CACHE, ('grin_s', 'grin_p', 'skk_s', 'skk_p'), _compute_R)
A = _cached(ABS_CACHE, ('askk_s', 'askk_p'), _compute_A)

# ---------------------------------------------------------------------------
# Shared reflectance colour scale across BOTH R rows and BOTH polarizations
# ---------------------------------------------------------------------------
rvals = np.concatenate([R[k].ravel() for k in ('skk_s', 'skk_p', 'grin_s', 'grin_p')])
rmin, rmax = rvals.min(), rvals.max()
vmin = 10.0 ** np.floor(np.log10(rmin))
vmax = 10.0 ** np.ceil(np.log10(rmax))
print(f"R range {rmin:.3e}..{rmax:.3e}  ->  colour limits {vmin:.0e}..{vmax:.0e}")
rnorm = LogNorm(vmin=vmin, vmax=vmax)

# ---------------------------------------------------------------------------
# Layout: 3 rows x 2 cols, square heatmaps, one shared colorbar per row
# ---------------------------------------------------------------------------
fig, axs, _ = C.panel_grid(nrows=3, ncols=2, panel_aspect=1.0)

ROWS = [
    (R['skk_s'], R['skk_p'], C.CMAP_RSKK, rnorm, r'$R_{\mathrm{sKK}}$'),
    (R['grin_s'], R['grin_p'], C.CMAP_RSKK, rnorm, r'$R_{\mathrm{GRIN}}$'),
    (A['askk_s'], A['askk_p'], C.CMAP_ASKK, None, r'$A_{\mathrm{sKK}}$'),
]

for r, (map_s, map_p, cmap, norm, cblabel) in enumerate(ROWS):
    for c, data in enumerate((map_s, map_p)):
        ax = axs[r, c]
        kw = dict(cmap=cmap, shading='auto')
        if norm is not None:
            kw['norm'] = norm
        m = ax.pcolormesh(ang, lol, data.T, **kw)
        ax.set_yscale('log')
        ax.set_ylim(LOL_MIN, LOL_MAX)
        ax.yaxis.set_major_locator(FixedLocator(LOL_TICKS))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if c == 0:                                   # left column carries the lambda axis
            ax.set_ylabel(r'$\lambda$')
            ax.yaxis.set_major_formatter(FixedFormatter(LOL_LABELS))
        else:
            ax.tick_params(labelleft=False)
        if r == len(ROWS) - 1:                       # bottom row carries the angle axis
            ax.set_xlabel(r'$\theta_b$ (deg)')
        else:
            ax.tick_params(labelbottom=False)
        if r == 0:                                   # polarization named once, at the top
            ax.set_title(('s' if c == 0 else 'p') + '-polarization')   # upright, not math italic
    # one bar per row, registered on the row's rightmost panel. Rows 0 and 1 use the SAME
    # norm, so bar 1 duplicates bar 0 on purpose -- it keeps the rows equal width.
    C.colorbar_strip(fig, axs[r, 1], m, cblabel, shared=True)

# Letters: relayout seats the LEFT column against its y-decoration (white margin, legible).
# The right column has no y-ticks, so its left edge IS the dark heatmap and a letter placed
# there is unreadable -- those three are placed by hand in the gutter instead, ha='right',
# the same treatment _mockup_fig3_4_combined.py gives its tick-less panel d.
C.relayout_grid(fig, axs, letters=['a', '', 'c', '', 'e', ''],
                panel_aspect=1.0, buffer_in=0.05, col_ws_in=0.30, row_ws_in=0.16)

_GUT = 0.012                                     # fig-fraction gap, letter -> panel left spine
for r, lab in enumerate(['b', 'd', 'f']):
    p = axs[r, 1].get_position()
    fig.text(p.x0 - _GUT, p.y1, r'$\mathbf{%s}$' % lab,
             fontsize=10, va='top', ha='right')

C.save(fig, os.path.join('theory', '_mockup_SI_1.png'))

# --- reported for the numerical-methods discussion: is the old 1e-8 floor real? ---
print(f"\ntrue minima  R_sKK  s={R['skk_s'].min():.3e}  p={R['skk_p'].min():.3e}")
print(f"             R_GRIN s={R['grin_s'].min():.3e}  p={R['grin_p'].min():.3e}")
print(f"maxima       A_sKK  s={A['askk_s'].max():.4f}  p={A['askk_p'].max():.4f}")
