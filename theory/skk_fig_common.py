"""skk_fig_common — shared conventions + helpers for the 2026-06 sKK figure overhaul.

This is a thin PRESENTATION layer on top of ``tmm_helper`` (the physics core). It does
NOT recompute any physics — it imports ``tmm_helper`` for that. Its job is to be the single
source of truth for the cross-cutting figure conventions in ``figure_plan.md`` so the
per-figure scripts (fig1_*, fig2_*, ...) stay consistent and don't drift.

Conventions encoded here (see figure_plan.md "Cross-cutting conventions"):
  * delta = 0.00125 primary (converged; monotonic R_sKK), 0.0025 fallback for colorplots
  * profiles plotted vs x/lambda (Horsley 2015); profile extent in x/lambda == d/lambda
  * theory angle axis = theta_b (angle in the nb background), range 0..theta_c ONLY
    (theta_c = arcsin(1/nb); never plot past it — beyond it is evanescent absorption, not sKK)
  * color = config/quantity; solid = s-pol, dashed = p-pol
  * profile colors: eps' blue, eps'' red, derivative quantities orange
  * TRA panels: T blue, R green, A red; y-label "Fraction of Power"
  * bold subplot letters OUTSIDE the axes box, no parentheses

Deferred (add when the consuming figure is ported, to avoid untested code):
  * d/lambda secondary top-axis helper (Fig 3+ 1D d/lambda plots)
  * distinct-colormap picks for colorplots (R_sKK viridis / ratio inferno / A_sKK cividis)
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import tmm_helper as tmm_h

# ---------------------------------------------------------------------------
# Physics / geometry conventions
# ---------------------------------------------------------------------------
NB = 1.7                      # non-dispersive background index (theory part)
DELTA = 0.00125               # primary discretization step (~1521 layers; converged)
DELTA_COLORPLOT = 0.0025      # fallback for colorplots (converged within trimmed range)
LAM_REF = 2.5                 # um — reference wavelength for the logistic theory figures
                              # (5 um coating -> d/lambda = 2; see figure_plan D5)

# Canonical FIGURE WIDTH = the printed text width, so \includegraphics[width=\linewidth]
# places the PNG ~1:1 (no rescale) and Python font points == on-page points. optica-article
# is one-column on letter paper with 1.625in L/R margins -> textwidth = 8.5 - 2*1.625 = 5.25in.
# Change ONLY this when switching journals (single/double column); aspect (height) is per-figure.
COL_WIDTH_IN = 5.25

# Data-panel width:height. WIDTH is fixed (column), so panel HEIGHT is derived from this
# ratio and total figure height grows as needed — a paper scrolls top-to-bottom, so height
# is the free dimension. Keeps panels from collapsing into ugly squares. House optics ratio.
PANEL_ASPECT = 4 / 3

# Deterministic inch-based layout reservations (see panel_grid). We DON'T use
# constrained_layout: it sizes axes to whatever's left after reserving label space, so it
# cannot guarantee a fixed 4:3 data box. Instead we place every axes at an explicit inch
# rectangle and reserve these fixed amounts of room for labels — exact aspect, deterministic
# label positions, and identical spacing across every figure (the consistency we want).
YLABEL_W_IN = 0.35     # horizontal room one y-axis needs (rotated label + tick numbers).
                       # MEASURED ~0.342 in for "Power Fraction" + "-1.0" ticks at these fonts.
XLABEL_H_IN = 0.29     # vertical room one x-axis needs (label + tick numbers). MEASURED ~0.279
                       # in (inward ticks add nothing outside) — must be ACCURATE or the row
                       # whitespace won't equal GAP_WS and the inter-row gaps look unequal.
LAYOUT_PAD_IN = 0.12   # outer breathing margin (top / right when no axis there)
CENTER_GAP_IN = 0.10   # whitespace between two columns' FACING y-labels (~3x the 2.5pt
                       # label<->tick gap, per Kunal) — tight column, labels don't collide
GAP_WS_IN = 0.18       # vertical whitespace below an x-axis label before the next row, AND
                       # below the schematic before the top row — panel_grid keeps them equal

THETA_C_DEG = float(np.degrees(np.arcsin(1.0 / NB)))   # ~36.03 deg for nb=1.7

# ---------------------------------------------------------------------------
# Canonical profile conventions  (LOCKED 2026-06-07)
# Domain is ALWAYS derived from the characteristic width — never hardcoded.
# See figure_plan.md "Profile conventions" and memory feedback_domain_tied_to_width.
# ---------------------------------------------------------------------------
WIDTH_LOGISTIC = 0.125 # um — logistic transition width w (a LENGTH, like gamma; = 1/k_old where
                       # k_old=8). Profile uses exp(x/w); domain = ±M_LOGISTIC*w -> d = 40w = 5 um.
M_LOGISTIC = 20        # logistic domain half-width in units of w (exp tail -> 20 saturates)
GAMMA_OVER_LAM = 0.1   # Lorentzian xi = gamma = 0.1*lambda (Horsley)
M_LORENTZ = 100        # Lorentzian domain half-width in units of gamma (1/x tail; 100 keeps
                       # the feature visible for an inset rather than a delta spike)
A_LORENTZ = 0.5        # Lorentzian amplitude

# ---------------------------------------------------------------------------
# Colors (figure_plan hexes — intentionally NOT colors.py, which uses a different palette)
# ---------------------------------------------------------------------------
C_EPS_RE = '#1f77b4'   # blue  — eps'(x), real part
C_EPS_IM = '#8B0000'   # red   — eps''(x), imaginary part
C_DERIV  = '#ff7f0e'   # orange — derivative quantities (deps'/dx, H[deps'/dx])
# TRA convention (one place T/R/A coloring applies)
C_T = '#1f77b4'        # blue
C_R = '#2ca02c'        # green
C_A = '#8B0000'        # red

# Line widths — sized for the TRUE printed width (COL_WIDTH_IN). At ~1.7in panels these give
# the same thin-curve look as the big (12in) prototypes do once squeezed into the column.
# (The old lw=2.5/2 were tuned for 11-12in figures and look ~3x too thick at 5.25in.)
LW_MAIN = 1.3          # primary curves (eps', eps'', T/R/A)
LW_THIN = 0.9          # secondary / dashed (intermediates, inset)
LW_GUIDE = 0.6         # zero/unity guide lines

FIGURES_DIR = os.path.join(_PROJECT_ROOT, 'sKK-Paper-Overleaf', 'figures')
THEORY_SCHEMATIC = os.path.join(FIGURES_DIR, 'theory_schematic.png')
APPLICATION_SCHEMATIC = os.path.join(FIGURES_DIR, 'application_schematic.png')  # TODO: PPT export

# Distinct colormaps per quantity (figure_plan convention #4 / D7)
CMAP_RSKK = 'viridis'    # absolute R_sKK
CMAP_RATIO = 'inferno'   # R_GRIN / R_sKK ratio (set white over-color for saturated wins)
CMAP_ASKK = 'cividis'    # absorption A_sKK


def apply_style():
    """Baseline rcParams shared by all overhaul figures.

    Figures are built at the true printed width (COL_WIDTH_IN), so point sizes here are the
    ON-PAGE sizes (no LaTeX rescale). ~8pt base is Optica-figure typical. These are global
    DEFAULTS — any single Artist can still be overridden per-call (e.g. ax.set_xlabel(...,
    fontsize=11)); rcParams just set the consistent baseline (point 6).

    NO constrained_layout: it can't hold a fixed panel aspect ratio (it sizes axes to fit
    labels). Layout is deterministic and inch-based instead — see panel_grid.

    Aesthetic conventions ported from Will Schmid's plot_functions.py (the group's
    publication style): Arial, INWARD ticks, minor ticks on, thin spines. Narrower font +
    inward ticks are why his y-labels hug the box.
    """
    plt.rcParams.update({
        # --- fonts (7pt base; narrow sans hugs the box). Per-Artist fontsize= still wins. ---
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        'axes.labelsize': 7.5,
        'axes.titlesize': 7.5,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        # --- legend: matplotlib DEFAULTS (soft ~0.8 gray edge, default 2.0 handle length,
        #     ROUNDED corners via fancybox) = the fig3 look Kunal liked. Only the font size
        #     is set: 0.67x the 7.5pt axis-label size (Will's legend:axis proportion) = 5pt.
        #     Do NOT re-override handlelength/edgecolor/fancybox. ---
        'legend.fontsize': 6,
        #'legend.fancybox': True,
        'legend.frameon': False,
        # --- consistent label<->tick spacing (point 4) ---
        'axes.labelpad': 2.5,
        'xtick.major.pad': 2.5,
        'ytick.major.pad': 2.5,
        # --- Will's publication look: inward ticks, minor ticks on, thin lines ---
        'axes.linewidth': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
        'xtick.minor.size': 2.0, 'ytick.minor.size': 2.0,
        # --- output ---
        'figure.dpi': 200,        # on-screen
        'savefig.dpi': 300,       # crisp export
    })


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def panel_grid(nrows, ncols, schem_panels=0.0, has_twin=False,
               panel_aspect=PANEL_ASPECT, width_in=COL_WIDTH_IN,
               margin_l=None, margin_r=None, margin_t=None, margin_b=None,
               col_gap=None, row_gap=None, schem_gap=None):
    """Deterministic inch-based grid of EXACTLY `panel_aspect` (4:3) data panels.

    Figure WIDTH is fixed to the column; panel width follows from it; panel height = width /
    aspect; total figure HEIGHT is whatever stacks those rows (a paper scrolls down, so
    height is free). Every axes is placed at an explicit inch rectangle via add_axes, so the
    4:3 box is exact and label positions are known without a draw-and-measure pass.

    Spacing is reserved in fixed inches (consistent across every figure, point 5):
      * `margin_l`  left y-axis room          (default YLABEL_W_IN)
      * `margin_b`  bottom x-axis room        (default XLABEL_H_IN)
      * `row_gap`   upper row's x-axis room + whitespace  (default XLABEL_H_IN + GAP_WS_IN)
      * `schem_gap` schematic<->top-row whitespace        (default GAP_WS_IN, == row_gap's
                    whitespace so both inter-row gaps read the same)
      * `col_gap`   right panel's y-axis room, PLUS the left panel's right twin-axis room
                    when `has_twin=True`, + CENTER_GAP_IN between the two facing y-labels
    Pass `has_twin=True` for figures with a secondary (right-side) y-axis so the column gap
    widens to clear it. Any reservation can be overridden per figure if a label is unusual.

    An optional schematic band spans all columns on top, height = schem_panels * panel_h.

    Returns (fig, axs, schem_ax):
      * axs      — np.ndarray of Axes, shape (nrows, ncols); unpack as (a, b), (c, d) = axs
      * schem_ax — the schematic Axes, or None when schem_panels == 0
    """
    margin_l = YLABEL_W_IN if margin_l is None else margin_l
    margin_r = LAYOUT_PAD_IN if margin_r is None else margin_r
    margin_t = LAYOUT_PAD_IN if margin_t is None else margin_t
    margin_b = XLABEL_H_IN if margin_b is None else margin_b
    if col_gap is None:
        # one y-axis block per facing side (+ the twin block on the left panel's right),
        # separated by CENTER_GAP_IN of whitespace between the two facing y-labels
        col_gap = YLABEL_W_IN + (YLABEL_W_IN if has_twin else 0.0) + CENTER_GAP_IN
    if row_gap is None:
        row_gap = XLABEL_H_IN + GAP_WS_IN      # upper row's x-axis room + whitespace below it
    if schem_gap is None:
        schem_gap = GAP_WS_IN                  # schematic has no x-label, so its whole gap is
                                               # whitespace -> schem<->top gap EQUALS the
                                               # x-label<->next-row whitespace (Kunal's #3)

    panel_w = (width_in - margin_l - margin_r - (ncols - 1) * col_gap) / ncols
    panel_h = panel_w / panel_aspect
    has_schem = schem_panels > 0
    schem_h = schem_panels * panel_h

    total_h = margin_t + margin_b + nrows * panel_h + (nrows - 1) * row_gap
    if has_schem:
        total_h += schem_h + schem_gap

    fig = plt.figure(figsize=(width_in, total_h))

    def _ax(x_in, top_in, w_in, h_in):
        """Add an axes from an inch rectangle whose TOP edge is `top_in` below the figure top."""
        return fig.add_axes([x_in / width_in,
                             (total_h - top_in - h_in) / total_h,
                             w_in / width_in, h_in / total_h])

    schem_ax = None
    y = margin_t                              # running offset from the figure TOP, in inches
    if has_schem:
        schem_ax = _ax(margin_l, y, width_in - margin_l - margin_r, schem_h)
        y += schem_h + schem_gap

    axs = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        x = margin_l
        for c in range(ncols):
            axs[r, c] = _ax(x, y, panel_w, panel_h)
            x += panel_w + col_gap
        y += panel_h + row_gap
    return fig, axs, schem_ax


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def theta_b_deg(step=0.5):
    """Theory incidence-angle axis: theta_b in [0, theta_c) degrees.

    arange stops strictly below theta_c (the TIR critical angle), per figure_plan
    ("do NOT plot past theta_c").
    """
    return np.arange(0.0, THETA_C_DEG, step)


def logistic_profile(w=WIDTH_LOGISTIC, nb=NB, npts=5001):
    """Canonical logistic eps(x) with exp(x/w), w = transition width (a LENGTH, like gamma).
    Domain = ±M_LOGISTIC*w (tied to width, NOT hardcoded) -> d = 40w. Returns (xx, e_re, e_im).
    tmm_h.logistic uses exp(k*x); we pass k = 1/w so exp(k*x) = exp(x/w) (byte-identical to k=1/w)."""
    xmax = M_LOGISTIC * w
    xx = np.linspace(-xmax, xmax, npts)
    e_re = tmm_h.logistic(xx, 1.0 / w, nb)
    e_im = tmm_h.ht_derivative(xx, e_re)
    return xx, e_re, e_im


def lorentzian_profile(lam=LAM_REF, a=A_LORENTZ, nb=NB, pts_per_gamma=100):
    """Canonical Lorentzian eps(x) = nb^2 - a*gam/(x+i*gam), xi=gam=GAMMA_OVER_LAM*lam (Horsley).
    Domain = ±M_LORENTZ*gam (tied to width; 1/x tail needs a large multiple). Returns (xx, ee, gam)."""
    gam = GAMMA_OVER_LAM * lam
    xmax = M_LORENTZ * gam
    npts = int(2 * xmax / (gam / pts_per_gamma)) + 1
    xx = np.linspace(-xmax, xmax, npts)
    ee = tmm_h.eps(xx, a, gam, nb)
    return xx, ee, gam


def build_stack(xx, ee, n_out, n_in=NB, delta=DELTA):
    """Discretize a continuous eps(x) profile into a TMM stack.

    Returns (n_list, d_list) framed as  n_in (semi-inf) | coating | n_out (semi-inf).
    Use n_out = 1.0 for the logistic (nb | coating | air); n_out = nb for the
    Lorentzian (nb | Lorentzian | nb, background on both sides).
    """
    nc, dc = tmm_h.discretize_profile(xx, ee, delta=delta)
    n_list = [n_in] + list(nc) + [n_out]
    d_list = [np.inf] + list(dc) + [np.inf]
    return n_list, d_list


def sweep_tra_vs_theta_b(n_list, d_list, angles_deg, lam=LAM_REF, pol='s'):
    """TMM TRA sweep with the angle interpreted DIRECTLY as theta_b (incidence in nb).

    No air->nb Snell conversion: in the isolated nb|coating|... geometry the launch
    medium IS nb, so theta_b is the honest variable (figure_plan angle convention).
    Returns (T, R, A) arrays.
    """
    T, R, A = [], [], []
    for ang in angles_deg:
        Ti, Ri, Ai = tmm_h.TRA(n_list, d_list, lamb=lam,
                               angle=np.radians(ang), pol=pol)
        T.append(Ti); R.append(Ri); A.append(Ai)
    return np.array(T), np.array(R), np.array(A)


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------
def label_panels(fig, pairs, dx_in=None, fontsize=10):
    """Bold panel letter at each panel's top-left corner, horizontally IN LINE with that
    panel's y-axis label (point 3). One rule for every panel: the letter's left edge sits
    `dx_in` inches left of the axes' left edge — the same place the y-label sits — so letter
    and y-label share an x and letters auto-align down the figure.

    Call ONCE, last. `pairs` = [(ax, 'a'), ...]. axes from panel_grid have explicit
    positions, so get_position() is exact with no draw needed. `dx_in` defaults to align
    with the y-label; nudge it if a letter overlaps the tick numbers.
    """
    dx_in = YLABEL_W_IN if dx_in is None else dx_in   # outer edge of the y-axis = where the
                                                      # y-label sits, so the letter lines up with it
    dx = dx_in / fig.get_size_inches()[0]
    for ax, letter in pairs:
        p = ax.get_position()
        fig.text(p.x0 - dx, p.y1, r'$\mathbf{%s}$' % letter,
                 fontsize=fontsize, va='top', ha='left')


def profile_panel(ax, x, e_re, e_im, ylab_im=r"$\epsilon''(x)$",
                  xlabel=r'$x/\lambda$', xlim=(-1, 1),
                  xticks=None, xticklabels=None, lw=LW_MAIN, im_ls='-'):
    """Dual-axis profile panel: eps'(x) (blue, left) and eps''(x)-like (red, right).

    `x` is plotted as given — pass x/lambda (default label) or physical x (Horsley
    convention: xlabel='x', xticks at multiples of lambda). `ylab_im` relabels the right
    axis (e.g. H[eps'] for the naive-HT panel). Returns the twin axis. Panel letters are
    placed separately by label_panels (this panel has a right twin -> use has_twin=True in
    panel_grid so the column gap clears the eps'' label).
    """
    axr = ax.twinx()
    ax.plot(x, e_re, color=C_EPS_RE, lw=lw)
    axr.plot(x, e_im, color=C_EPS_IM, lw=lw, ls=im_ls)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\epsilon'(x)$", color=C_EPS_RE)
    axr.set_ylabel(ylab_im, color=C_EPS_IM)
    ax.tick_params(axis='y', labelcolor=C_EPS_RE)
    axr.tick_params(axis='y', labelcolor=C_EPS_IM)
    axr.yaxis.set_minor_locator(AutoMinorLocator(2))   # one minor between majors on 2nd y-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))   # one minor between majors on y-axis
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
    return axr


def tra_panel(ax, angles_deg, T, R, A, mark_gain=False, legend=True,
              ylim=None, xlabel=r'$\theta_b$ (deg)', xmax=THETA_C_DEG):
    """TRA-vs-angle panel. T blue, R green, A red. Panel letters via label_panels.

    Defaults to the theory cone (theta_b, 0..theta_c). The Lorentzian has NO air interface
    (eps_b on both sides) -> no TIR -> pass xlabel='Angle of incidence (deg)', xmax=90 to
    span all angles.
    """
    ax.plot(angles_deg, T, color=C_T, lw=LW_MAIN, label='T')   # upright (Arial), not $T$ italic
    ax.plot(angles_deg, R, color=C_R, lw=LW_MAIN, label='R')
    ax.plot(angles_deg, A, color=C_A, lw=LW_MAIN, label='A')
    if mark_gain:
        ax.axhline(0, color='gray', lw=LW_GUIDE, ls=':')
        ax.axhline(1, color='gray', lw=LW_GUIDE, ls=':')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Power Fraction')
    ax.set_xlim(0, xmax)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend()   # matplotlib-default frame (soft gray, rounded) — see apply_style


def sweep_tra_vs_wavelength(n_list, d_list, lams, angle_deg=0.0, pol='s'):
    """TMM TRA vs wavelength at fixed theta_b. Returns (T, R, A) arrays.

    For the non-dispersive theory geometry, sweeping lambda at a fixed coating is the
    cleanest way to traverse the d/lambda axis (scale invariance): d/lambda = d_total/lams.
    """
    T, R, A = [], [], []
    for wl in lams:
        Ti, Ri, Ai = tmm_h.TRA(n_list, d_list, lamb=wl,
                               angle=np.radians(angle_deg), pol=pol)
        T.append(Ti); R.append(Ri); A.append(Ai)
    return np.array(T), np.array(R), np.array(A)


def tra_map(n_list, d_list, angles_deg, lams, pol='s'):
    """2D TMM sweep over (theta_b, lambda). Returns R, A arrays shaped (n_angles, n_lams)."""
    R = np.zeros((len(angles_deg), len(lams)))
    A = np.zeros((len(angles_deg), len(lams)))
    for i, ang in enumerate(angles_deg):
        for j, wl in enumerate(lams):
            _, Rij, Aij = tmm_h.TRA(n_list, d_list, lamb=wl,
                                    angle=np.radians(ang), pol=pol)
            R[i, j] = Rij; A[i, j] = Aij
    return R, A


def coating_thickness(d_list):
    """Total coating thickness (sum of finite interior layer thicknesses)."""
    return float(np.sum([d for d in d_list if np.isfinite(d)]))


def dlambda_top_axis(ax, d_um, label=r'$\lambda$ ($\mu$m)'):
    """Add a secondary TOP axis converting the bottom d/lambda axis to physical wavelength
    for one reference coating thickness d_um (figure_plan convention #2). Bottom axis must
    already be d/lambda. lambda = d_um / (d/lambda)."""
    def fwd(dl):
        dl = np.asarray(dl, dtype=float)
        out = np.full_like(dl, np.nan)
        nz = dl != 0
        out[nz] = d_um / dl[nz]
        return out
    sec = ax.secondary_xaxis('top', functions=(fwd, fwd))  # involutive: lambda=d/(d/lam)
    sec.set_xlabel(label)
    return sec


def colorbar_in_panel(fig, ax, mappable, label,
                      cbar_w_in=0.08, gap_in=0.06, labelroom_in=0.42):
    """Attach a vertical colorbar by carving it from the RIGHT EDGE of ax's own box.

    panel_grid hands each colorplot the same fixed 4:3 rectangle it hands a line panel. A
    plain fig.colorbar(ax=...) would either steal width from OUTSIDE that box (breaking the
    deterministic inch layout) or bleed the colorbar label into the page margin. Instead we
    SHRINK the heatmap and drop the colorbar (+ its ticks + label) into a reserved strip on
    the right, so the WHOLE panel — heatmap + gap + colorbar + label — stays exactly the 4:3
    rectangle (Kunal's colorplot spec: the panel incl. colorbar is 4:3, and whatever width is
    left after the colorbar becomes the data grid). The heatmap ends up a touch narrower than
    a line panel; that's intended. `reserve = cbar_w + gap + labelroom`, all in inches,
    converted to figure fraction. ax.get_position().x0 (left edge) is unchanged, so a later
    label_panels still aligns the panel letter with the y-label.
    """
    W = fig.get_size_inches()[0]
    pos = ax.get_position()
    reserve = (cbar_w_in + gap_in + labelroom_in) / W
    ax.set_position([pos.x0, pos.y0, pos.width - reserve, pos.height])
    cax = fig.add_axes([pos.x0 + pos.width - reserve + gap_in / W, pos.y0,
                        cbar_w_in / W, pos.height])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label)
    cb.outline.set_linewidth(0.6)
    cax.tick_params(width=0.6)
    return cb


def save(fig, path, dpi=300):
    """Save and report. `path` may be relative to the project root.

    NO bbox_inches='tight': panel_grid already fixes a deterministic, minimal margin, and
    tight-bbox would re-crop the canvas after layout — rescaling the whole figure and
    breaking the COL_WIDTH_IN-sized fonts/line widths (and shifting the panel letters off
    their axes). Report the true saved size as a sanity check that figsize survived.
    """
    if not os.path.isabs(path):
        path = os.path.join(_PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi)            # NO bbox_inches='tight' — it breaks sizing
    print('saved', path, fig.get_size_inches())
    return path
