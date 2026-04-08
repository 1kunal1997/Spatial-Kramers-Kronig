"""
Spatial KK Analysis — Consolidated Script
==========================================
All calculations and figure generation for the sKK AR coating paper.

Usage:
  # Run all figures:
  python theory/skk_analysis_consolidated.py

  # Run specific figures by name:
  python theory/skk_analysis_consolidated.py fig1 fig6 task1

  # List available figure names:
  python theory/skk_analysis_consolidated.py --list

  # Save to a custom output directory:
  python theory/skk_analysis_consolidated.py --outdir sKK-Paper-Overleaf/figures fig6 fig7

  # Interactive mode (VS Code #%% cells):
  #   Run the "Setup" cell first, then run any individual figure cell.

Available figures:
  fig1  — Lorentzian HT + GRIN periodic continuation (2-panel)
  fig2  — Asymmetric endpoint problem (single curve: direct FFT)
  fig3  — Derivative-then-integrate result (no textbox in a, lossy air in b)
  fig4  — Backside reflection: sKK vs GRIN vs Bulk
  fig5  — Spectral FoM introduction (single panel)
  fig6  — R-A tradeoff with spectral FoM on twin axis
  fig7  — Sigma gating: profiles + reflection (no bulk curve)
  fig8  — FoM comparison: full sKK vs gated (both panels)
  fig9  — Angle-resolved backside reflection
  fig10 — Thickness design space (single panel)
  loss_shapes       — Loss shape comparison (Batch 1 + Batch 2)
  width_amplitude   — Width-amplitude tradeoff (thin coating)
  thick_shapes      — Thick coating loss shapes + width-amplitude
  task1 — R_back 2D colorplots (angle x wavelength, 4 thicknesses, 2 pols)
  task2 — Thickness sweep for all loss shapes (4 parameter combos)
  task3 — Losses-matched thin vs thick comparison

Requirements:
  - numpy, scipy, matplotlib, tmm
  - Sapphire data: lam_um_T_K_Al2O3_no_ko_ne_ke.dat

NOTE: This script re-implements profile generation locally for self-contained
paper figure reproduction. FoM functions and spectral FoM plotting use
tmm_helper (skk_spectral_fom, hilbert_fom_derivative, plot_spectral_fom).
TMM calculations use tmm_helper.TRA() for automatic coherent/incoherent
layer classification.
"""

import sys, os
from types import SimpleNamespace
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import tmm
import tmm_helper as tmm_h
from scipy.signal import hilbert  # used directly in Figs 2-3 (naive HT demonstration)
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# ============================================================================
# Plot style
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 200, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'axes.linewidth': 1.2,
    'lines.linewidth': 2.0, 'mathtext.fontset': 'cm',
})

BLUE = '#1f77b4'
RED = '#8B0000'
GREEN = '#2ca02c'
ORANGE = '#ff7f0e'
PURPLE = '#9467bd'

# ============================================================================
# Sapphire optical constants (realistic mid-IR)
# ============================================================================
def load_sapphire_data():
    """Load sapphire ordinary-ray optical constants (2-5 um)."""
    ri_path = os.path.join(_PROJECT_ROOT, 'RI', 'lam_um_T_K_Al2O3_no_ko_ne_ke.dat')
    for ri_path in [ri_path]:
        if os.path.exists(ri_path):
            data = np.genfromtxt(ri_path)
            # columns: lam(um), T(K), n_o, k_o, n_e, k_e
            kdata = data[50:351, 3]
            ndata = data[50:351, 2]
            lamdata = data[50:351, 0]
            print(f"Loaded sapphire data: lam = {lamdata[0]:.2f}-{lamdata[-1]:.2f} um, "
                  f"{len(lamdata)} pts, n = {ndata.min():.4f}-{ndata.max():.4f}")
            return lamdata, ndata, kdata
    raise FileNotFoundError("Sapphire data file not found")


# Core physics functions: all from tmm_helper
# logistic_eps  -> tmm_h.logistic()
# eps_lorentz   -> tmm_h.eps()
# ht_derivative -> tmm_h.ht_derivative()
# smooth_gate   -> tmm_h.smooth_gate()
# discretize_profile -> tmm_h.discretize_profile()
# skk_spectral_fom   -> tmm_h.skk_spectral_fom()
# hilbert_fom        -> tmm_h.hilbert_fom_derivative()

# ============================================================================
# TMM helpers
# ============================================================================
def Rback_vs_wavelength(n_coating, d_coating, ndata, kdata, lamdata, angle_deg, pol):
    """Backside reflection and absorption vs wavelength (auto-coherence)."""
    deg = np.pi/180; angle = angle_deg * deg
    Rb = np.zeros(len(lamdata)); At = np.zeros(len(lamdata))
    for i, wl in enumerate(lamdata):
        n_sub = complex(ndata[i], kdata[i])
        n_t = [1, n_sub] + list(n_coating) + [1]
        d_t = [np.inf, 5000] + list(d_coating) + [np.inf]
        T, R, A = tmm_h.TRA(n_t, d_t, lamb=wl, angle=angle, pol=pol)
        th_f = tmm.snell(1, n_sub, angle)
        Rf = tmm.interface_R(pol, 1, n_sub, angle, th_f)
        Rb[i] = R - Rf
        At[i] = A
    return Rb, At

def Rback_vs_angle(n_coating, d_coating, n_sub, angle_list_deg, lam, pol):
    """Backside reflection vs angle at a single wavelength (auto-coherence)."""
    deg = np.pi/180
    n_t = [1, n_sub] + list(n_coating) + [1]
    d_t = [np.inf, 5000] + list(d_coating) + [np.inf]
    Rb = np.zeros(len(angle_list_deg)); At = np.zeros(len(angle_list_deg))
    for i, ang in enumerate(angle_list_deg):
        theta = ang * deg
        T, R, A = tmm_h.TRA(n_t, d_t, lamb=lam, angle=theta, pol=pol)
        th_f = tmm.snell(1, n_sub, theta)
        Rf = tmm.interface_R(pol, 1, n_sub, theta, th_f)
        Rb[i] = R - Rf
        At[i] = A
    return Rb, At

def Rback_bulk_wl(ndata, kdata, lamdata, angle_deg, pol):
    """Bulk sapphire backside reflection (no coating, auto-coherence)."""
    deg = np.pi/180; angle = angle_deg * deg
    Rb = np.zeros(len(lamdata)); At = np.zeros(len(lamdata))
    for i, wl in enumerate(lamdata):
        n_sub = complex(ndata[i], kdata[i])
        n_b = [1, n_sub, 1]
        d_b = [np.inf, 5000, np.inf]
        T, R, A = tmm_h.TRA(n_b, d_b, lamb=wl, angle=angle, pol=pol)
        th_f = tmm.snell(1, n_sub, angle)
        Rf = tmm.interface_R(pol, 1, n_sub, angle, th_f)
        Rb[i] = R - Rf; At[i] = A
    return Rb, At

def Rback_bulk_angle(n_sub, angle_list_deg, lam, pol):
    """Bulk sapphire backside reflection vs angle (auto-coherence)."""
    deg = np.pi/180
    n_b = [1, n_sub, 1]; d_b = [np.inf, 5000, np.inf]
    Rb = np.zeros(len(angle_list_deg))
    for i, ang in enumerate(angle_list_deg):
        theta = ang * deg
        T, R, A = tmm_h.TRA(n_b, d_b, lamb=lam, angle=theta, pol=pol)
        th_f = tmm.snell(1, n_sub, theta)
        Rf = tmm.interface_R(pol, 1, n_sub, theta, th_f)
        Rb[i] = R - Rf
    return Rb

def _find_contiguous(xx, mask):
    """Find contiguous True regions in a boolean mask."""
    regions = []
    in_region = False
    for i in range(len(mask)):
        if mask[i] and not in_region:
            start = xx[i]; in_region = True
        elif not mask[i] and in_region:
            regions.append((start, xx[i])); in_region = False
    if in_region:
        regions.append((start, xx[-1]))
    return regions


def Rback_2D(n_coating, d_coating, ndata, kdata, lamdata, angle_list_deg, pol):
    """R_back and absorption on a 2D (angle x wavelength) grid (auto-coherence)."""
    deg = np.pi / 180
    Rb = np.zeros((len(angle_list_deg), len(lamdata)))
    At = np.zeros((len(angle_list_deg), len(lamdata)))
    for i, ang in enumerate(angle_list_deg):
        angle = ang * deg
        for j, wl in enumerate(lamdata):
            n_sub = complex(ndata[j], kdata[j])
            n_t = [1, n_sub] + list(n_coating) + [1]
            d_t = [np.inf, 5000] + list(d_coating) + [np.inf]
            T, R, A = tmm_h.TRA(n_t, d_t, lamb=wl, angle=angle, pol=pol)
            th_f = tmm.snell(1, n_sub, angle)
            Rf = tmm.interface_R(pol, 1, n_sub, angle, th_f)
            Rb[i, j] = R - Rf
            At[i, j] = A
    return Rb, At


def Rback_bulk_2D(ndata, kdata, lamdata, angle_list_deg, pol):
    """Bulk sapphire R_back on a 2D (angle x wavelength) grid (no coating)."""
    deg = np.pi / 180
    Rb = np.zeros((len(angle_list_deg), len(lamdata)))
    for i, ang in enumerate(angle_list_deg):
        angle = ang * deg
        for j, wl in enumerate(lamdata):
            n_sub = complex(ndata[j], kdata[j])
            n_b = [1, n_sub, 1]
            d_b = [np.inf, 5000, np.inf]
            T, R, A = tmm_h.TRA(n_b, d_b, lamb=wl, angle=angle, pol=pol)
            th_f = tmm.snell(1, n_sub, angle)
            Rf = tmm.interface_R(pol, 1, n_sub, angle, th_f)
            Rb[i, j] = R - Rf
    return Rb


def _annotate_geomean(ax, data_2d):
    """Annotate a colorplot subplot with the geometric mean of the data."""
    pos = data_2d[data_2d > 0]
    if len(pos):
        gm = np.exp(np.mean(np.log(pos)))
        ax.text(0.97, 0.03, f'\u27e8\u00b7\u27e9={gm:.1e}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))


def _spectrum_physical(x, u, v):
    """Derivative → FT → ÷ik with physical dx normalization.

    Multiplies FFT by dx so |ε̂(k)|² has physical units matching exact
    analytic curves (C/sinh², C·exp). Only needed for crossover overlay plots;
    for FoM percentages use skk_spectral_fom() directly (normalization cancels).
    Returns (k, pwr).
    """
    x = np.asarray(x, float)
    ud = np.gradient(np.asarray(u, float), x)
    vd = np.gradient(np.asarray(v, float), x)
    z_d = ud + 1j * vd
    dx = x[1] - x[0]
    Z_d = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(z_d))) * dx
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(z_d), d=dx))
    eps_hat = np.zeros_like(Z_d)
    nonzero = k != 0
    eps_hat[nonzero] = Z_d[nonzero] / (1j * k[nonzero])
    eps_hat[~nonzero] = 0
    return k, np.abs(eps_hat)**2


def _direct_ft_fom(x, u, v):
    """Plain FT of ε(x) = u+iv (no derivative, no ÷ik) → FoM, k, |ε̂(k)|².

    Used to demonstrate why the derivative pre-processing is necessary:
    without it the DC spike from non-zero endpoints contaminates the FoM.
    """
    x = np.asarray(x, float)
    z = np.asarray(u, float) + 1j * np.asarray(v, float)
    dx = x[1] - x[0]
    Z = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(z)))
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(z), d=dx))
    pwr = np.abs(Z)**2
    E_pos = np.sum(pwr[k > 0])
    E_neg = np.sum(pwr[k < 0])
    fom = 100 * max(0.0, (E_pos - E_neg) / (E_pos + E_neg))
    return fom, k, pwr


# ============================================================================
# Loss shape profile generators
# ============================================================================
def make_constant_profile(xx, target_integral):
    """Constant epsilon'' everywhere, scaled to match target integral."""
    L = xx[-1] - xx[0]
    c = target_integral / L
    return np.full_like(xx, max(c, 0.0))

def make_gaussian_profile(xx, target_integral):
    """Gaussian peaked at center, width ~ coating extent / 4."""
    sigma_g = (xx[-1] - xx[0]) / 8
    g = np.exp(-xx**2 / (2 * sigma_g**2))
    integral = np.trapezoid(g, xx)
    if integral > 0:
        g *= target_integral / integral
    return np.maximum(g, 0.0)

def make_double_peak_profile(xx, target_integral):
    """Two Gaussians at +/- L/4."""
    L = xx[-1] - xx[0]
    sigma_g = L / 16
    x1, x2 = -L/4, L/4
    g = np.exp(-(xx - x1)**2 / (2*sigma_g**2)) + np.exp(-(xx - x2)**2 / (2*sigma_g**2))
    integral = np.trapezoid(g, xx)
    if integral > 0:
        g *= target_integral / integral
    return np.maximum(g, 0.0)

def make_random_profile(xx, target_integral, seed=42):
    """Smooth random profile (filtered noise), scaled to match integral."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(xx))
    sigma_filt = len(xx) // 20
    smooth = gaussian_filter1d(noise, sigma_filt)
    smooth -= smooth.min()  # ensure non-negative
    integral = np.trapezoid(smooth, xx)
    if integral > 0:
        smooth *= target_integral / integral
    return smooth


def scale_eim(xx, eim_ref, x_c, s, target_loss):
    """Scale epsilon'' width by factor s around x_c, then rescale to preserve integral."""
    interp = interp1d(xx, eim_ref, kind='cubic', bounds_error=False, fill_value=0.0)
    x_scaled = x_c + (xx - x_c) / s
    eim_scaled = interp(x_scaled) / s
    current_loss = np.trapezoid(eim_scaled, xx)
    if current_loss > 1e-15:
        eim_scaled *= target_loss / current_loss
    return eim_scaled


def plot_shape_figure(xx, e_re, e_im_shape, shape_name, fname,
                      ndata, kdata, lamdata, angle_deg, pol, delta, figdir):
    """Plot a two-panel figure: profile (left) + R_back/A vs wavelength (right)."""
    ee = e_re + 1j * e_im_shape
    nc, dc = tmm_h.discretize_profile(xx, ee, delta=delta)
    Rb, At = Rback_vs_wavelength(nc, dc, ndata, kdata, lamdata, angle_deg, pol)

    R_avg = np.trapezoid(Rb, lamdata) / (lamdata[-1] - lamdata[0])
    A_avg = np.trapezoid(At, lamdata) / (lamdata[-1] - lamdata[0])
    sf = tmm_h.skk_spectral_fom(xx, e_re, e_im_shape)[0]
    total_loss = np.trapezoid(e_im_shape, xx)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: epsilon'(x) and epsilon''(x)
    ax_l2 = ax_l.twinx()
    ax_l.plot(xx, e_re, color=BLUE, lw=2.5)
    ax_l2.plot(xx, e_im_shape, color=RED, lw=2.5)
    ax_l.set_xlabel(r'$x$ ($\mu$m)')
    ax_l.set_ylabel(r"$\epsilon'$", color=BLUE)
    ax_l2.set_ylabel(r"$\epsilon''$", color=RED)
    ax_l.tick_params(axis='y', labelcolor=BLUE)
    ax_l2.tick_params(axis='y', labelcolor=RED)
    ax_l.set_title(f'{shape_name}')
    eim_max = np.max(e_im_shape)
    eim_min = np.min(e_im_shape)
    ax_l2.set_ylim(min(0, eim_min * 1.1), eim_max * 1.6)
    ax_l.text(0.97, 0.96, f'Spectral FoM: {sf:.1f}%',
              transform=ax_l.transAxes, fontsize=10, va='top', ha='right',
              bbox=dict(facecolor='wheat', alpha=0.8, boxstyle='round,pad=0.4'))

    # Right panel: R_back and A vs wavelength
    ax_r.plot(lamdata, Rb, color=GREEN, lw=2.5, label=r'$R_{\rm back}$')
    ax_r.plot(lamdata, At, color=RED, lw=2.5, label='$A$')
    ax_r.set_xlabel(r'Wavelength ($\mu$m)')
    ax_r.set_ylabel('Fraction of Power')
    ax_r.set_title(f'$\\langle R \\rangle$={R_avg:.4f}, $\\langle A \\rangle$={A_avg:.4f}')
    ax_r.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{figdir}/{fname}')
    plt.close()
    return R_avg, A_avg, sf, total_loss


# ============================================================================
# Setup — shared data for all figures
# ============================================================================
def setup(figdir=None):
    """Compute shared parameters, grids, profiles, and TMM baselines.

    Returns a SimpleNamespace S with all shared data.
    """
    S = SimpleNamespace()

    # Output directory
    if figdir is None:
        S.FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    else:
        S.FIGDIR = os.path.join(_PROJECT_ROOT, figdir) if not os.path.isabs(figdir) else figdir
    os.makedirs(S.FIGDIR, exist_ok=True)

    # ---- Core parameters ----
    S.k_steep = 100; S.nb = 1.7; S.delta = 0.01
    dx = 1/(100*S.k_steep)
    S.xmin = -20/S.k_steep; S.xmax = -S.xmin
    nx = 1 + int(np.floor((S.xmax - S.xmin) / dx))
    S.xx = np.linspace(S.xmin, S.xmax, nx)
    S.e_re = tmm_h.logistic(S.xx, S.k_steep, S.nb)

    # Sapphire data
    S.lamdata, S.ndata, S.kdata = load_sapphire_data()

    # Derivative HT
    S.e_im_deriv = tmm_h.ht_derivative(S.xx, S.e_re)

    S.angle_test = 80; S.pol_test = 's'

    # ---- Coating stacks ----
    ee_full = S.e_re + 1j * S.e_im_deriv
    S.nc_full, S.dc_full = tmm_h.discretize_profile(S.xx, ee_full, delta=S.delta)
    ee_grin = S.e_re + 0j
    S.nc_grin, S.dc_grin = tmm_h.discretize_profile(S.xx, ee_grin, delta=S.delta)

    # ---- TMM baseline results ----
    S.Rb_full, S.A_full = Rback_vs_wavelength(S.nc_full, S.dc_full, S.ndata, S.kdata,
                                               S.lamdata, S.angle_test, S.pol_test)
    S.Rb_grin, S.A_grin = Rback_vs_wavelength(S.nc_grin, S.dc_grin, S.ndata, S.kdata,
                                               S.lamdata, S.angle_test, S.pol_test)
    S.Rb_bulk, S.A_bulk = Rback_bulk_wl(S.ndata, S.kdata, S.lamdata, S.angle_test, S.pol_test)

    S.nk_d = np.sqrt(S.e_re + 1j * S.e_im_deriv)
    S.mask_lossy = (np.real(S.nk_d) < 1.15) & (np.imag(S.nk_d) > 0.01)

    print("=== FoM Comparison ===")
    for name, eim in [("Derivative", S.e_im_deriv)]:
        hf = tmm_h.hilbert_fom_derivative(S.xx, S.e_re, eim)[0]
        sf = tmm_h.skk_spectral_fom(S.xx, S.e_re, eim)[0]
        print(f"  {name}: HT_FoM={hf:.2f}%, Spectral_FoM={sf:.2f}%")

    S.Rb_avg_bulk = np.trapezoid(S.Rb_bulk, S.lamdata) / (S.lamdata[-1] - S.lamdata[0])
    S.Rb_avg_grin = np.trapezoid(S.Rb_grin, S.lamdata) / (S.lamdata[-1] - S.lamdata[0])
    S.Rb_avg_full = np.trapezoid(S.Rb_full, S.lamdata) / (S.lamdata[-1] - S.lamdata[0])
    S.A_avg_full = np.trapezoid(S.A_full, S.lamdata) / (S.lamdata[-1] - S.lamdata[0])
    print(f"\n=== Wavelength sweep ({S.pol_test}-pol, {S.angle_test} deg) ===")
    print(f"  Bulk:     <R_back> = {S.Rb_avg_bulk:.5f}")
    print(f"  GRIN:     <R_back> = {S.Rb_avg_grin:.5f}")
    print(f"  sKK full: <R_back> = {S.Rb_avg_full:.5f}, <A> = {S.A_avg_full:.5f}")

    # ---- M=2000 domain for accurate spectral FoM (figs 5, 8) ----
    M_fom = 2000
    dx_fom = 1 / (100 * S.k_steep)
    xmin_fom = -M_fom / S.k_steep
    S.xx_fom = np.linspace(xmin_fom, -xmin_fom,
                           1 + int(np.floor((-2 * xmin_fom) / dx_fom)))
    S.e_re_fom = (S.nb**2 - 1) / (1 + np.exp(S.k_steep * S.xx_fom)) + 1
    S.e_im_fom = tmm_h.ht_derivative(S.xx_fom, S.e_re_fom)

    # ---- Dense grid for loss shape comparison (M=20) ----
    S.xx_dense = np.linspace(-0.2, 0.2, 4001)
    S.e_re_dense = (S.nb**2 - 1) / (1 + np.exp(S.k_steep * S.xx_dense)) + 1
    S.e_im_dense = tmm_h.ht_derivative(S.xx_dense, S.e_re_dense)

    # ---- Thick coating grid (k_steep=10, ~4 um) ----
    S.k_steep_thick = 10
    dx_thick = 1 / (100 * S.k_steep_thick)
    xmin_thick = -20 / S.k_steep_thick
    xmax_thick = -xmin_thick
    S.xx_thick = np.linspace(xmin_thick, xmax_thick,
                             1 + int(np.floor((xmax_thick - xmin_thick) / dx_thick)))
    S.e_re_thick = tmm_h.logistic(S.xx_thick, S.k_steep_thick, S.nb)
    S.e_im_thick = tmm_h.ht_derivative(S.xx_thick, S.e_re_thick)

    return S


# ============================================================================
# Figure functions
# ============================================================================

def fig_lorentz_vs_grin(S):
    """Figure 1: Lorentzian HT + GRIN periodic continuation (2-panel)."""
    A_lor = 0.5; x0_lor = 0.05
    dx_l = x0_lor / 50; xmax_l = x0_lor * 100
    nx_l = 1 + int(np.floor(2 * xmax_l / dx_l))
    xx_l = np.linspace(-xmax_l, xmax_l, nx_l)
    ee_l = tmm_h.eps(xx_l, A_lor, x0_lor, S.nb)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Lorentzian profile
    ax1b = ax1.twinx()
    ax1.plot(xx_l, ee_l.real, color=BLUE, lw=2.5)
    ax1b.plot(xx_l, ee_l.imag, color=RED, lw=2.5)
    ax1.set_xlabel(r'$x$ ($\mu$m)'); ax1.set_ylabel(r"$\epsilon'$", color=BLUE)
    ax1b.set_ylabel(r"$\epsilon''$", color=RED)
    ax1.set_title(r'(a) Lorentzian: $\epsilon(x) = n_b^2 - \frac{A \cdot x_0}{x + i \cdot x_0}$')
    ax1.tick_params(axis='y', labelcolor=BLUE)
    ax1b.tick_params(axis='y', labelcolor=RED)
    ax1.text(0.03, 0.70, 'Same endpoints:\n'
             + r"$\epsilon'(-\infty) = \epsilon'(+\infty) = n_b^2$" + '\n'
             + r'$\Rightarrow$ FFT works perfectly',
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round'))

    # (b) GRIN periodic continuation -> jump discontinuity
    e_tiled = np.tile(S.e_re, 3)
    xx_tiled = np.concatenate([S.xx - (S.xmax-S.xmin), S.xx, S.xx + (S.xmax-S.xmin)])
    ax2.plot(xx_tiled, e_tiled, color=BLUE, lw=2)
    ax2.axvline(S.xmin, color='gray', lw=1, ls='--', alpha=0.5)
    ax2.axvline(S.xmax, color='gray', lw=1, ls='--', alpha=0.5)
    ax2.axvspan(S.xmin, S.xmax, alpha=0.06, color='blue')
    ax2.text((S.xmin+S.xmax)/2, 2.4, 'One period', fontsize=10, ha='center', color=BLUE,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor=BLUE, boxstyle='round'))
    ax2.annotate('Jump\ndiscontinuity', xy=(S.xmax, 1.5),
                 xytext=(S.xmax + 0.12, 2.0), fontsize=10, ha='center',
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
                 color=RED, bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round'))
    ax2.set_xlabel(r'$x$ ($\mu$m)'); ax2.set_ylabel(r"$\epsilon'(x)$")
    ax2.set_title(r"(b) GRIN: FFT periodic continuation $\to$ jump discontinuity")
    ax2.set_xlim(S.xmin - 0.25, S.xmax + 0.35)

    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_new1_lorentz_vs_grin.png')
    plt.close()
    print("Saved fig 1: Lorentzian vs GRIN")


def fig_endpoint_problem(S):
    """Figure 2: Asymmetric endpoint problem — single curve."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) GRIN profile with endpoint labels
    ax = axes[0]
    ax.plot(S.xx, S.e_re, color=BLUE, lw=2.5)
    ax.axhline(S.nb**2, color='gray', lw=1, ls=':', alpha=0.5)
    ax.axhline(1.0, color='gray', lw=1, ls=':', alpha=0.5)
    ax.annotate(r'$\epsilon_b = n_b^2 = %.2f$' % S.nb**2, xy=(S.xmin, S.nb**2),
                xytext=(S.xmin + 0.02, S.nb**2 + 0.08), fontsize=11, color='gray')
    ax.annotate(r'$\epsilon_{\rm air} = 1$', xy=(S.xmax, 1.0),
                xytext=(S.xmax - 0.08, 1.15), fontsize=11, color='gray')
    ax.annotate('', xy=(0.15, 1.0), xytext=(0.15, S.nb**2),
                arrowprops=dict(arrowstyle='<->', color=RED, lw=2))
    ax.text(0.165, (1 + S.nb**2)/2, r'$\Delta\epsilon = %.2f$' % (S.nb**2 - 1),
            fontsize=12, color=RED, va='center')
    ax.set_xlabel(r'$x$ ($\mu$m)'); ax.set_ylabel(r"$\epsilon'(x)$")
    ax.set_title(r"(a) GRIN profile: different endpoints ($\epsilon_b \neq \epsilon_{\rm air}$)")

    # (b) Direct FFT HT — single orange curve only
    ax = axes[1]
    z_naive = hilbert(S.e_re)
    e_im_naive = np.imag(z_naive)
    ax.plot(S.xx, e_im_naive, color=ORANGE, lw=2, label='No padding (direct FFT)')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.annotate('Edge artifacts from\nendpoint mismatch',
                xy=(S.xmin + 0.02, e_im_naive[10]), xytext=(0.0, max(e_im_naive)*0.8),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round'))
    ax.set_xlabel(r'$x$ ($\mu$m)'); ax.set_ylabel(r"$\epsilon''(x)$")
    ax.set_title(r"(b) Naive Hilbert transform — endpoint artifacts")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig0a_ht_problem.png')
    plt.close()
    print("Saved fig 2: Asymmetric endpoint problem")


def fig_derivative_result(S):
    """Figure 3: Derivative-then-integrate result."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Key insight: derivatives -> 0 at both ends
    ax = axes[0]; ax_twin = ax.twinx()
    u_deriv = np.gradient(S.e_re, S.xx)
    pad_n = 8 * len(S.xx)
    v_deriv_ht = np.imag(hilbert(np.pad(u_deriv, (pad_n, pad_n),
                                         mode='constant')))[pad_n:pad_n+len(S.xx)]
    ax.plot(S.xx, S.e_re, color=BLUE, lw=2.5, label=r"$\epsilon'(x)$")
    ax.plot(S.xx, u_deriv, color=ORANGE, lw=2, label=r"$d\epsilon'/dx$")
    ax_twin.plot(S.xx, v_deriv_ht, color=RED, lw=2, label=r"$\mathcal{H}[d\epsilon'/dx]$")
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r"$\epsilon'$, $d\epsilon'/dx$", color=BLUE)
    ax_twin.set_ylabel(r"$\mathcal{H}[d\epsilon'/dx]$", color=RED)
    ax.set_title(r"(a) Key insight: $d\epsilon'/dx \to 0$ at both ends")
    ax.tick_params(axis='y', labelcolor=BLUE)
    ax_twin.tick_params(axis='y', labelcolor=RED)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=9)

    # (b) Result: epsilon' and epsilon'' with lossy air shaded
    ax = axes[1]; ax2 = ax.twinx()
    ax.plot(S.xx, S.e_re, color=BLUE, lw=2.5)
    ax2.plot(S.xx, S.e_im_deriv, color=RED, lw=2.5)
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r"$\epsilon'$", color=BLUE)
    ax2.set_ylabel(r"$\epsilon''$", color=RED)
    ax.set_title(r"(b) Result: $\epsilon(x) = \epsilon' + i\epsilon''$ (derivative method)")
    ax.tick_params(axis='y', labelcolor=BLUE)
    ax2.tick_params(axis='y', labelcolor=RED)

    lossy_mask_b = (S.e_re < 1.1) & (S.e_im_deriv > 0.01)
    if np.any(lossy_mask_b):
        idx_start = np.where(lossy_mask_b)[0][0]
        idx_end = np.where(lossy_mask_b)[0][-1]
        ax.axvspan(S.xx[idx_start], S.xx[idx_end], alpha=0.15, color='orange', zorder=0)
        y_mid = 0.75 * ax.get_ylim()[1] + 0.25 * ax.get_ylim()[0]
        x_mid = 0.5 * (S.xx[idx_start] + S.xx[idx_end])
        ax.text(x_mid, y_mid, '"Lossy air"\n' + r'$n \approx 1, k > 0$',
                fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='wheat', alpha=0.8, boxstyle='round'))

    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_new3_derivative_result.png')
    plt.close()
    print("Saved fig 3: Derivative result")


def fig_reflection_wavelength(S):
    """Figure 4: Backside reflection comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(S.lamdata, S.Rb_bulk, color='gray', lw=2, label='Bulk sapphire')
    ax.plot(S.lamdata, S.Rb_grin, '--', color=BLUE, lw=2, label='GRIN coating')
    ax.plot(S.lamdata, S.Rb_full, color=GREEN, lw=2.5, label='sKK coating')
    ax.set_xlabel(r'Wavelength ($\mu$m)'); ax.set_ylabel(r'$R_{\rm back}$')
    ax.set_title(f'(a) Backside reflection ({S.pol_test}-pol, {S.angle_test}\u00b0)')
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.plot(S.lamdata, S.A_bulk, color='gray', lw=2, label='Bulk sapphire')
    ax.plot(S.lamdata, S.A_grin, '--', color=BLUE, lw=2, label='GRIN coating')
    ax.plot(S.lamdata, S.A_full, color=RED, lw=2.5, label='sKK coating')
    ax.set_xlabel(r'Wavelength ($\mu$m)'); ax.set_ylabel('Absorbance')
    ax.set_title(f'(b) Absorption ({S.pol_test}-pol, {S.angle_test}\u00b0)')
    ax.legend(fontsize=10, loc='center left')

    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig3_reflection_wavelength.png')
    plt.close()
    print("Saved fig 4: Backside reflection")


def fig_fom_intro(S):
    """Figure 5: Spectral FoM introduction — single panel."""
    fom_full, k_freq, power_d = tmm_h.skk_spectral_fom(
        S.xx_fom, S.e_re_fom, S.e_im_fom)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))
    tmm_h.plot_spectral_fom(ax, k_freq, power_d, fom_full,
                            title=f'Full sKK profile \u2014 Spectral FoM = {fom_full:.1f}%')
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_fom_intro_single.png')
    plt.close()
    print("Saved fig 5: FoM intro (single panel)")


def fig_alpha_tradeoff(S):
    """Figure 6: R-A tradeoff with spectral FoM on twin axis."""
    alpha_list = np.linspace(0, 1, 40)
    R_vs_a = np.zeros(len(alpha_list))
    A_vs_a = np.zeros(len(alpha_list))
    FoM_vs_a = np.zeros(len(alpha_list))

    for j, alpha in enumerate(alpha_list):
        e_im_a = alpha * S.e_im_deriv
        ee_a = S.e_re + 1j * e_im_a
        nc_a, dc_a = tmm_h.discretize_profile(S.xx, ee_a, delta=S.delta)
        Rb_a, At_a = Rback_vs_wavelength(nc_a, dc_a, S.ndata, S.kdata, S.lamdata,
                                          S.angle_test, S.pol_test)
        R_vs_a[j] = np.trapezoid(Rb_a, S.lamdata) / (S.lamdata[-1] - S.lamdata[0])
        A_vs_a[j] = np.trapezoid(At_a, S.lamdata) / (S.lamdata[-1] - S.lamdata[0])
        FoM_vs_a[j] = tmm_h.skk_spectral_fom(S.xx, S.e_re, e_im_a)[0] / 100.0
        if j % 10 == 0:
            print(f"  alpha={alpha:.2f}: <R>={R_vs_a[j]:.4f}, <A>={A_vs_a[j]:.4f}, FoM={FoM_vs_a[j]:.3f}")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax2 = ax.twinx()
    ax.plot(alpha_list, R_vs_a, color=GREEN, lw=2.5, label=r'$\langle R_{\rm back} \rangle_\lambda$')
    ax.plot(alpha_list, A_vs_a, color=RED, lw=2.5, label=r'$\langle A \rangle_\lambda$')
    ax2.plot(alpha_list, FoM_vs_a, '--', color=PURPLE, lw=2, label='Spectral FoM')
    ax.set_xlabel(r'$\alpha$ (imaginary scaling factor)', fontsize=14)
    ax.set_ylabel('Fraction of Power', fontsize=14)
    ax2.set_ylabel('Spectral FoM', color=PURPLE, fontsize=14)
    ax2.tick_params(axis='y', labelcolor=PURPLE)
    ax2.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc='center left')
    ax2.legend(fontsize=11, loc='center right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig4_alpha_tradeoff.png')
    plt.close()
    print("Saved fig 6: Alpha tradeoff")


def fig_sigma_gating(S):
    """Figure 7: Sigma gating — profiles + reflection (no bulk curve)."""
    sigma_list = [None, 0.1, 0.01]
    n0_gate = 1.3

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    fig.subplots_adjust(hspace=0.35, wspace=0.45)

    for idx, sigma in enumerate(sigma_list):
        e_im_base = S.e_im_deriv.copy()
        if sigma is not None:
            gate = tmm_h.smooth_gate(S.e_re, n0_gate**2, sigma)
            e_im_gated = e_im_base * gate
            label = f'$\\sigma$ = {sigma}'
        else:
            e_im_gated = e_im_base
            label = r'No gating ($\alpha=1$)'

        # Top row: dielectric function
        ax = axes[0, idx]; ax2t = ax.twinx()
        ax.plot(S.xx, S.e_re, color=BLUE, lw=2.5)
        ax2t.plot(S.xx, e_im_gated, color=RED, lw=2.5)
        ax.set_xlabel(r'$x$ ($\mu$m)'); ax.set_ylabel(r"$\epsilon'$", color=BLUE)
        ax2t.set_ylabel(r"$\epsilon''$", color=RED)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelcolor=BLUE)
        ax2t.tick_params(axis='y', labelcolor=RED)
        sf = tmm_h.skk_spectral_fom(S.xx, S.e_re, e_im_gated)[0]
        ax.text(0.97, 0.97, f'Spectral FoM:\n{sf:.1f}%',
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(facecolor='wheat', alpha=0.7, boxstyle='round'))

        # Bottom row: reflection (no bulk curve)
        ax = axes[1, idx]
        ee_g = S.e_re + 1j * e_im_gated
        nc_g, dc_g = tmm_h.discretize_profile(S.xx, ee_g, delta=S.delta)
        Rb_g, At_g = Rback_vs_wavelength(nc_g, dc_g, S.ndata, S.kdata, S.lamdata,
                                          S.angle_test, S.pol_test)
        ax.plot(S.lamdata, Rb_g, color=GREEN, lw=2.5, label=r'$R_{\rm back}$')
        ax.plot(S.lamdata, At_g, color=RED, lw=2.5, label='$A$')
        ax.set_xlabel(r'Wavelength ($\mu$m)'); ax.set_ylabel('Fraction of Power')
        R_avg = np.trapezoid(Rb_g, S.lamdata)/(S.lamdata[-1]-S.lamdata[0])
        A_avg = np.trapezoid(At_g, S.lamdata)/(S.lamdata[-1]-S.lamdata[0])
        ax.set_title(f'$\\langle R \\rangle$={R_avg:.4f}, $\\langle A \\rangle$={A_avg:.4f}')
        ax.legend(fontsize=9)

    plt.savefig(f'{S.FIGDIR}/fig5_sigma_gating.png')
    plt.close()
    print("Saved fig 7: Sigma gating")


def fig_fom_comparison(S):
    """Figure 8: FoM comparison — full sKK vs gated (both panels)."""
    fom_full, k_freq, power_d = tmm_h.skk_spectral_fom(
        S.xx_fom, S.e_re_fom, S.e_im_fom)

    gate_01_fom = tmm_h.smooth_gate(S.e_re_fom, 1.3**2, 0.1)
    e_im_gated_fom = S.e_im_fom * gate_01_fom

    fom_gated, k_g, power_g = tmm_h.skk_spectral_fom(
        S.xx_fom, S.e_re_fom, e_im_gated_fom)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    tmm_h.plot_spectral_fom(ax1, k_freq, power_d, fom_full,
                            title=f'(a) Full sKK profile \u2014 Spectral FoM = {fom_full:.1f}%')
    tmm_h.plot_spectral_fom(ax2, k_g, power_g, fom_gated,
                            title=r'(b) Gated ($\sigma=0.1$)' + f' \u2014 Spectral FoM = {fom_gated:.1f}%')
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_new5_fom_explanation.png')
    plt.close()
    print("Saved fig 8: FoM comparison")


def fig_angle_resolved(S):
    """Figure 9: Angle-resolved backside reflection."""
    angle_list = np.arange(0, 90, 1)
    lam_test = 3.0
    idx_lam = np.argmin(np.abs(S.lamdata - lam_test))
    n_sub = complex(S.ndata[idx_lam], S.kdata[idx_lam])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for pidx, pol in enumerate(['s', 'p']):
        ax = axes[pidx]
        Rb_bulk_a = Rback_bulk_angle(n_sub, angle_list, lam_test, pol)
        Rb_grin_a, _ = Rback_vs_angle(S.nc_grin, S.dc_grin, n_sub, angle_list, lam_test, pol)
        Rb_skk_a, _ = Rback_vs_angle(S.nc_full, S.dc_full, n_sub, angle_list, lam_test, pol)
        ax.plot(angle_list, Rb_bulk_a, color='gray', lw=2, label='Bulk sapphire')
        ax.plot(angle_list, Rb_grin_a, '--', color=BLUE, lw=2, label='GRIN coating')
        ax.plot(angle_list, Rb_skk_a, color=GREEN, lw=2.5, label='sKK coating')
        ax.set_xlabel('Angle of Incidence (\u00b0)')
        ax.set_ylabel(r'$R_{\rm back}$')
        ax.set_title(f'{pol}-polarization, $\\lambda$ = {lam_test} $\\mu$m')
        ax.legend(fontsize=10); ax.set_xlim(0, 89)
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig6_angle_resolved.png')
    plt.close()
    print("Saved fig 9: Angle-resolved")


def fig_thickness_single(S):
    """Figure 10: Thickness design space — single panel."""
    k_values = np.array([2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100])
    thicknesses = []
    R_grin_arr, R_skk_arr, R_gated_arr = [], [], []
    A_skk_arr = []

    for k_val in k_values:
        dx_k = 1/(100*k_val); xmin_k = -20/k_val; xmax_k = -xmin_k
        nx_k = 1+int(np.floor((xmax_k-xmin_k)/dx_k))
        xx_k = np.linspace(xmin_k, xmax_k, nx_k)
        e_re_k = tmm_h.logistic(xx_k, k_val, S.nb)
        e_im_k = tmm_h.ht_derivative(xx_k, e_re_k)
        thickness = 2*xmax_k
        thicknesses.append(thickness)

        # GRIN
        nc_gk, dc_gk = tmm_h.discretize_profile(xx_k, e_re_k + 0j, delta=S.delta)
        Rb_gk, _ = Rback_vs_wavelength(nc_gk, dc_gk, S.ndata, S.kdata, S.lamdata,
                                        S.angle_test, S.pol_test)
        R_grin_arr.append(np.trapezoid(Rb_gk, S.lamdata) / (S.lamdata[-1] - S.lamdata[0]))

        # Full sKK
        nc_fk, dc_fk = tmm_h.discretize_profile(xx_k, e_re_k + 1j*e_im_k, delta=S.delta)
        Rb_fk, At_fk = Rback_vs_wavelength(nc_fk, dc_fk, S.ndata, S.kdata, S.lamdata,
                                             S.angle_test, S.pol_test)
        R_skk_arr.append(np.trapezoid(Rb_fk, S.lamdata) / (S.lamdata[-1] - S.lamdata[0]))
        A_skk_arr.append(np.trapezoid(At_fk, S.lamdata) / (S.lamdata[-1] - S.lamdata[0]))

        # Gated sKK (sigma=0.1)
        gate_k = tmm_h.smooth_gate(e_re_k, 1.3**2, 0.1)
        nc_gatk, dc_gatk = tmm_h.discretize_profile(xx_k, e_re_k + 1j*e_im_k*gate_k, delta=S.delta)
        Rb_gatk, _ = Rback_vs_wavelength(nc_gatk, dc_gatk, S.ndata, S.kdata, S.lamdata,
                                          S.angle_test, S.pol_test)
        R_gated_arr.append(np.trapezoid(Rb_gatk, S.lamdata) / (S.lamdata[-1] - S.lamdata[0]))

        print(f"  k={k_val:3d} ({thickness:.1f} um): R_GRIN={R_grin_arr[-1]:.4f}, "
              f"R_sKK={R_skk_arr[-1]:.4f}, R_gated={R_gated_arr[-1]:.4f}")

    thicknesses = np.array(thicknesses)
    R_grin_arr = np.array(R_grin_arr)
    R_skk_arr = np.array(R_skk_arr)
    R_gated_arr = np.array(R_gated_arr)
    A_skk_arr = np.array(A_skk_arr)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    ax.axhline(S.Rb_avg_bulk, color='gray', lw=1.5, ls=':', label='Bulk sapphire', zorder=1)
    ax.plot(thicknesses, R_grin_arr, 'o-', color=BLUE, lw=2.5, ms=7, label='GRIN coating')
    ax.plot(thicknesses, R_skk_arr, 's-', color=GREEN, lw=2.5, ms=7, label='sKK coating')
    ax.plot(thicknesses, R_gated_arr, '^-', color=PURPLE, lw=2.5, ms=7,
            label=r'sKK gated ($\sigma$=0.1)')
    ax.set_xlabel(r'Coating thickness ($\mu$m)')
    ax.set_ylabel(r'$\langle R_{\rm back} \rangle$')
    ax.set_title(r'Backside reflection vs coating thickness (s-pol, 80\u00b0, $\lambda$-averaged 2\u20135 $\mu$m)')
    ax.set_xscale('log'); ax.set_xlim(0.3, 25)

    # Second y-axis for sKK absorption
    ax2 = ax.twinx()
    ax2.plot(thicknesses, A_skk_arr, 's--', color=RED, lw=2, ms=6, alpha=0.6,
             label=r'sKK absorption')
    ax2.set_ylabel(r'$\langle A \rangle$ (sKK coating)', color=RED)
    ax2.tick_params(axis='y', labelcolor=RED)

    # Combined legend from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

    # Shaded regime regions
    ax.axvspan(0.3, 1.5, alpha=0.08, color='red', zorder=0)
    ax.axvspan(1.5, 6, alpha=0.08, color='yellow', zorder=0)
    ax.axvspan(6, 25, alpha=0.08, color='green', zorder=0)
    ax.text(0.7, 0.02, r'Sub-$\lambda$' + '\n(sKK regime)', fontsize=9,
            ha='center', va='bottom', color='darkred')
    ax.text(3.0, 0.02, 'Crossover', fontsize=9, ha='center', va='bottom', color='olive')
    ax.text(12, 0.02, 'GRIN\nsufficient', fontsize=9, ha='center', va='bottom', color='darkgreen')

    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig9_thickness_single.png')
    plt.close()
    print("Saved fig 10: Thickness design space")


def fig_loss_shapes(S):
    """Loss shape comparison: Batch 1 (unconstrained) + Batch 2 (gated)."""
    shape_names = ['sKK (HT derivative)', 'Constant', 'Gaussian', 'Double peaks', 'Random']
    shape_fnames = ['skk', 'constant', 'gaussian', 'double', 'random']

    # --- Batch 1: Unconstrained loss placement ---
    print("\n=== LOSS SHAPE COMPARISON - Batch 1: Unconstrained ===")
    loss_ref = np.trapezoid(S.e_im_dense, S.xx_dense)
    print(f"  Reference total loss (sKK): int(eps'')dx = {loss_ref:.6f}")

    shapes_b1 = [
        np.copy(S.e_im_dense),
        make_constant_profile(S.xx_dense, loss_ref),
        make_gaussian_profile(S.xx_dense, loss_ref),
        make_double_peak_profile(S.xx_dense, loss_ref),
        make_random_profile(S.xx_dense, loss_ref),
    ]

    _col = 'int(e")'
    print(f"  {'Shape':<22s} {_col:>10s} {'<R>':>10s} {'<A>':>10s} {'FoM%':>8s}")
    print(f"  {'-'*54}")
    for name, fname, e_im_shape in zip(shape_names, shape_fnames, shapes_b1):
        R_avg, A_avg, sf, tl = plot_shape_figure(
            S.xx_dense, S.e_re_dense, e_im_shape, name,
            f'fig_batch1_{fname}.png', S.ndata, S.kdata, S.lamdata,
            S.angle_test, S.pol_test, S.delta, S.FIGDIR)
        print(f"  {name:<22s} {tl:10.6f} {R_avg:10.5f} {A_avg:10.5f} {sf:8.1f}")
        print(f"    Saved fig_batch1_{fname}.png")

    # --- Batch 2: Gated loss placement ---
    print("\n=== LOSS SHAPE COMPARISON - Batch 2: Gated ===")
    gate_dense = tmm_h.smooth_gate(S.e_re_dense, 1.3**2, 0.1)
    e_im_gated_ref = S.e_im_dense * gate_dense
    loss_ref_gated = np.trapezoid(e_im_gated_ref, S.xx_dense)
    print(f"  Reference total loss (gated sKK): int(eps'')dx = {loss_ref_gated:.6f}")

    shapes_b2 = []
    for e_im_shape in shapes_b1:
        gated = e_im_shape * gate_dense
        gated_integral = np.trapezoid(gated, S.xx_dense)
        if gated_integral > 1e-15:
            gated *= loss_ref_gated / gated_integral
        shapes_b2.append(gated)

    _col = 'int(e")'
    print(f"  {'Shape':<22s} {_col:>10s} {'<R>':>10s} {'<A>':>10s} {'FoM%':>8s}")
    print(f"  {'-'*54}")
    for name, fname, e_im_shape in zip(shape_names, shape_fnames, shapes_b2):
        R_avg, A_avg, sf, tl = plot_shape_figure(
            S.xx_dense, S.e_re_dense, e_im_shape, name,
            f'fig_batch2_{fname}.png', S.ndata, S.kdata, S.lamdata,
            S.angle_test, S.pol_test, S.delta, S.FIGDIR)
        print(f"  {name:<22s} {tl:10.6f} {R_avg:10.5f} {A_avg:10.5f} {sf:8.1f}")
        print(f"    Saved fig_batch2_{fname}.png")



def fig_thick_shapes(S):
    """Thick coating loss shapes + width-amplitude (k_steep=10, ~4 um)."""
    shape_names = ['sKK (HT derivative)', 'Constant', 'Gaussian', 'Double peaks', 'Random']
    shape_fnames = ['skk', 'constant', 'gaussian', 'double', 'random']

    print(f"\n=== THICKER COATING - LOSS SHAPE COMPARISON ===")
    print(f"  k_steep = {S.k_steep_thick}, grid: {S.xx_thick[0]:.1f} to {S.xx_thick[-1]:.1f} um, "
          f"{len(S.xx_thick)} pts")
    print(f"  Coating thickness ~ {S.xx_thick[-1] - S.xx_thick[0]:.1f} um")

    loss_ref_thick = np.trapezoid(S.e_im_thick, S.xx_thick)
    print(f"  Reference total loss (sKK): int(eps'')dx = {loss_ref_thick:.6f}")

    shapes_thick = [
        np.copy(S.e_im_thick),
        make_constant_profile(S.xx_thick, loss_ref_thick),
        make_gaussian_profile(S.xx_thick, loss_ref_thick),
        make_double_peak_profile(S.xx_thick, loss_ref_thick),
        make_random_profile(S.xx_thick, loss_ref_thick),
    ]

    _col = 'int(e")'
    print(f"  {'Shape':<22s} {_col:>10s} {'<R>':>10s} {'<A>':>10s} {'FoM%':>8s}")
    print(f"  {'-'*54}")
    for name, fname, e_im_shape in zip(shape_names, shape_fnames, shapes_thick):
        R_avg, A_avg, sf, tl = plot_shape_figure(
            S.xx_thick, S.e_re_thick, e_im_shape, name,
            f'fig_batch1_thick_{fname}.png', S.ndata, S.kdata, S.lamdata,
            S.angle_test, S.pol_test, S.delta, S.FIGDIR)
        print(f"  {name:<22s} {tl:10.6f} {R_avg:10.5f} {A_avg:10.5f} {sf:8.1f}")
        print(f"    Saved fig_batch1_thick_{fname}.png")

    # ---- Width-amplitude tradeoff (thick) ----
    print("\n=== WIDTH-AMPLITUDE TRADEOFF - Thick Coating (k_steep=10) ===")

    x_center_thick = S.xx_thick[np.argmax(S.e_im_thick)]
    print(f"  eps'' peak center: x_c = {x_center_thick:.4f} um")

    loss_ref_width_thick = np.trapezoid(S.e_im_thick, S.xx_thick)

    # --- Figure 1: Overlay of 5 eps'' curves (broader discrete values) ---
    s_discrete_thick = [0.25, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0]
    colors_s_thick = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for s_val, col in zip(s_discrete_thick, colors_s_thick):
        eim_s = scale_eim(S.xx_thick, S.e_im_thick, x_center_thick, s_val, loss_ref_width_thick)
        label = f'$s$ = {s_val}'
        if s_val == 1.0:
            label += ' (reference)'
        ax.plot(S.xx_thick, eim_s, color=col, lw=2.5 if s_val == 1.0 else 1.8, label=label)
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r"$\epsilon''(x)$")
    ax.legend(fontsize=10)
    ax.set_xlim(-1.5, 1.5)
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_width_thick_profiles.png')
    plt.close()
    print("  Saved fig_width_thick_profiles.png")

    # Verify total loss is constant
    for s_val in s_discrete_thick:
        eim_s = scale_eim(S.xx_thick, S.e_im_thick, x_center_thick, s_val, loss_ref_width_thick)
        loss_s = np.trapezoid(eim_s, S.xx_thick)
        print(f"    s={s_val:.2f}: int(eps'')dx = {loss_s:.6f}")

    # --- Figure 2: <R> and <A> vs s (continuous sweep, broader range) ---
    s_sweep_thick = np.linspace(0.1, 10.0, 80)
    R_vs_s_thick = np.zeros(len(s_sweep_thick))
    A_vs_s_thick = np.zeros(len(s_sweep_thick))

    for j, s_val in enumerate(s_sweep_thick):
        eim_s = scale_eim(S.xx_thick, S.e_im_thick, x_center_thick, s_val, loss_ref_width_thick)
        ee_s = S.e_re_thick + 1j * eim_s
        nc_s, dc_s = tmm_h.discretize_profile(S.xx_thick, ee_s, delta=S.delta)
        Rb_s, At_s = Rback_vs_wavelength(nc_s, dc_s, S.ndata, S.kdata, S.lamdata,
                                          S.angle_test, S.pol_test)
        R_vs_s_thick[j] = np.trapezoid(Rb_s, S.lamdata) / (S.lamdata[-1] - S.lamdata[0])
        A_vs_s_thick[j] = np.trapezoid(At_s, S.lamdata) / (S.lamdata[-1] - S.lamdata[0])
        if (j + 1) % 10 == 0:
            print(f"    s sweep (thick): {j+1}/{len(s_sweep_thick)}")

    fig, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: Absorption (linear scale)
    ax_a.plot(s_sweep_thick, A_vs_s_thick, color=RED, lw=2.5)
    R_interp_thick = np.interp(np.array(s_discrete_thick), s_sweep_thick, R_vs_s_thick)
    A_interp_thick = np.interp(np.array(s_discrete_thick), s_sweep_thick, A_vs_s_thick)
    for i, (s_val, col) in enumerate(zip(s_discrete_thick, colors_s_thick)):
        ax_a.plot(s_val, A_interp_thick[i], 's', color=col, ms=9, zorder=5,
                  markeredgecolor='black', markeredgewidth=1.0)
    ax_a.set_xlabel(r'Width scaling factor $s$', fontsize=14)
    ax_a.set_ylabel(r'$\langle A \rangle_\lambda$', fontsize=14)
    ax_a.grid(True, alpha=0.3)

    # Right panel: R_back (log scale to show minimum at s=1)
    ax_r.plot(s_sweep_thick, R_vs_s_thick, color=GREEN, lw=2.5)
    for i, (s_val, col) in enumerate(zip(s_discrete_thick, colors_s_thick)):
        ax_r.plot(s_val, R_interp_thick[i], 'o', color=col, ms=9, zorder=5,
                  markeredgecolor='black', markeredgewidth=1.0)
    ax_r.set_yscale('log')
    ax_r.set_xlabel(r'Width scaling factor $s$', fontsize=14)
    ax_r.set_ylabel(r'$\langle R_{\rm back} \rangle_\lambda$', fontsize=14)
    ax_r.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_width_thick_sweep.png')
    plt.close()
    print("  Saved fig_width_thick_sweep.png")

    # Print values at discrete s
    print(f"\n  {'s':>5s}  {'<R>':>10s}  {'<A>':>10s}")
    print(f"  {'-'*30}")
    for s_val in s_discrete_thick:
        idx = np.argmin(np.abs(s_sweep_thick - s_val))
        print(f"  {s_val:5.2f}  {R_vs_s_thick[idx]:10.5f}  {A_vs_s_thick[idx]:10.5f}")


def fig_task1_colorplots(S):
    """Task 1: R_back 2D colorplots (angle x wavelength, 4 thicknesses, 2 pols)."""
    print("\n=== TASK 1: 2D COLORPLOTS ===")

    k_color_vals = [80, 40, 8, 4]
    color_angle_list = np.arange(0, 90, 1)
    color_pols = ['s', 'p']

    # Pre-compute bulk reference (pol-dependent but coating-independent)
    Rb_bulk_2D = {}
    for pol_c in color_pols:
        print(f"  Computing bulk 2D ({pol_c}-pol)...")
        Rb_bulk_2D[pol_c] = Rback_bulk_2D(S.ndata, S.kdata, S.lamdata, color_angle_list, pol_c)

    # For each thickness, compute sKK and GRIN 2D arrays
    Rb_skk_2D = {pol_c: [] for pol_c in color_pols}
    Rb_grin_2D = {pol_c: [] for pol_c in color_pols}
    color_thicknesses = []

    for k_c in k_color_vals:
        dx_c = 1 / (100 * k_c); xmin_c = -20 / k_c; xmax_c = -xmin_c
        nx_c = 1 + int(np.floor((xmax_c - xmin_c) / dx_c))
        xx_c = np.linspace(xmin_c, xmax_c, nx_c)
        e_re_c = tmm_h.logistic(xx_c, k_c, S.nb)
        e_im_c = tmm_h.ht_derivative(xx_c, e_re_c)
        thickness_c = xmax_c - xmin_c
        color_thicknesses.append(thickness_c)

        nc_skk_c, dc_skk_c = tmm_h.discretize_profile(xx_c, e_re_c + 1j * e_im_c, delta=S.delta)
        nc_grin_c, dc_grin_c = tmm_h.discretize_profile(xx_c, e_re_c + 0j, delta=S.delta)

        for pol_c in color_pols:
            print(f"  k={k_c} ({thickness_c:.1f} um), {pol_c}-pol: computing sKK 2D...")
            Rb_s, _ = Rback_2D(nc_skk_c, dc_skk_c, S.ndata, S.kdata, S.lamdata,
                               color_angle_list, pol_c)
            Rb_skk_2D[pol_c].append(Rb_s)
            print(f"  k={k_c} ({thickness_c:.1f} um), {pol_c}-pol: computing GRIN 2D...")
            Rb_g, _ = Rback_2D(nc_grin_c, dc_grin_c, S.ndata, S.kdata, S.lamdata,
                               color_angle_list, pol_c)
            Rb_grin_2D[pol_c].append(Rb_g)

    # ---- Figure A: R_back sKK colorplots (2 pols x 4 thicknesses) ----
    fig_A, axes_A = plt.subplots(2, 4, figsize=(20, 8))
    for col, (k_c, thick_c) in enumerate(zip(k_color_vals, color_thicknesses)):
        for row, pol_c in enumerate(color_pols):
            ax = axes_A[row, col]
            data = Rb_skk_2D[pol_c][col]
            im = ax.pcolormesh(color_angle_list, S.lamdata, data.T,
                               norm=matplotlib.colors.LogNorm(vmin=1e-5, vmax=0.2),
                               cmap='viridis', shading='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xlabel('AoI (degrees)')
            ax.set_ylabel(r'Wavelength ($\mu$m)')
            ax.set_title(f'R_back,sKK \u2014 {thick_c:.1f} \u03bcm, {pol_c}-pol')
            _annotate_geomean(ax, data)
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_colorplot_Rback_sKK.png', dpi=150)
    plt.close()
    print("  Saved fig_colorplot_Rback_sKK.png")

    # ---- Figure B: R_back GRIN colorplots ----
    fig_B, axes_B = plt.subplots(2, 4, figsize=(20, 8))
    for col, (k_c, thick_c) in enumerate(zip(k_color_vals, color_thicknesses)):
        for row, pol_c in enumerate(color_pols):
            ax = axes_B[row, col]
            data = Rb_grin_2D[pol_c][col]
            im = ax.pcolormesh(color_angle_list, S.lamdata, data.T,
                               norm=matplotlib.colors.LogNorm(vmin=1e-5, vmax=0.2),
                               cmap='viridis', shading='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xlabel('AoI (degrees)')
            ax.set_ylabel(r'Wavelength ($\mu$m)')
            ax.set_title(f'R_back,GRIN \u2014 {thick_c:.1f} \u03bcm, {pol_c}-pol')
            _annotate_geomean(ax, data)
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_colorplot_Rback_GRIN.png', dpi=150)
    plt.close()
    print("  Saved fig_colorplot_Rback_GRIN.png")

    # ---- Figure C: R_bulk / R_sKK ratio ----
    fig_C, axes_C = plt.subplots(2, 4, figsize=(20, 8))
    for col, (k_c, thick_c) in enumerate(zip(k_color_vals, color_thicknesses)):
        for row, pol_c in enumerate(color_pols):
            ax = axes_C[row, col]
            ratio = Rb_bulk_2D[pol_c] / np.clip(Rb_skk_2D[pol_c][col], 1e-10, None)
            im = ax.pcolormesh(color_angle_list, S.lamdata, ratio.T,
                               norm=matplotlib.colors.LogNorm(vmin=1, vmax=1e4),
                               cmap='viridis', shading='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xlabel('AoI (degrees)')
            ax.set_ylabel(r'Wavelength ($\mu$m)')
            ax.set_title(f'R_bulk/R_sKK \u2014 {thick_c:.1f} \u03bcm, {pol_c}-pol')
            _annotate_geomean(ax, ratio)
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_colorplot_ratio_bulk_over_sKK.png', dpi=150)
    plt.close()
    print("  Saved fig_colorplot_ratio_bulk_over_sKK.png")

    # ---- Figure D: R_GRIN / R_sKK ratio ----
    fig_D, axes_D = plt.subplots(2, 4, figsize=(20, 8))
    for col, (k_c, thick_c) in enumerate(zip(k_color_vals, color_thicknesses)):
        for row, pol_c in enumerate(color_pols):
            ax = axes_D[row, col]
            ratio = Rb_grin_2D[pol_c][col] / np.clip(Rb_skk_2D[pol_c][col], 1e-10, None)
            im = ax.pcolormesh(color_angle_list, S.lamdata, ratio.T,
                               norm=matplotlib.colors.LogNorm(vmin=1, vmax=1e4),
                               cmap='viridis', shading='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xlabel('AoI (degrees)')
            ax.set_ylabel(r'Wavelength ($\mu$m)')
            ax.set_title(f'R_GRIN/R_sKK \u2014 {thick_c:.1f} \u03bcm, {pol_c}-pol')
            _annotate_geomean(ax, ratio)
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_colorplot_ratio_GRIN_over_sKK.png', dpi=150)
    plt.close()
    print("  Saved fig_colorplot_ratio_GRIN_over_sKK.png")

    # ---- Figure E: R_bulk (no coating) colorplots ----
    fig_E, axes_E = plt.subplots(1, 2, figsize=(12, 4))
    for col, pol_c in enumerate(color_pols):
        ax = axes_E[col]
        data = Rb_bulk_2D[pol_c]
        im = ax.pcolormesh(color_angle_list, S.lamdata, data.T,
                           norm=matplotlib.colors.LogNorm(vmin=1e-5, vmax=0.2),
                           cmap='viridis', shading='auto')
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('AoI (degrees)')
        ax.set_ylabel(r'Wavelength ($\mu$m)')
        ax.set_title(f'R_bulk \u2014 {pol_c}-pol')
        _annotate_geomean(ax, data)
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_colorplot_Rback_bulk.png', dpi=150)
    plt.close()
    print("  Saved fig_colorplot_Rback_bulk.png")

    # ---- Figure F: R_bulk / R_GRIN ratio ----
    fig_F, axes_F = plt.subplots(2, 4, figsize=(20, 8))
    for col, (k_c, thick_c) in enumerate(zip(k_color_vals, color_thicknesses)):
        for row, pol_c in enumerate(color_pols):
            ax = axes_F[row, col]
            ratio = Rb_bulk_2D[pol_c] / np.clip(Rb_grin_2D[pol_c][col], 1e-10, None)
            im = ax.pcolormesh(color_angle_list, S.lamdata, ratio.T,
                               norm=matplotlib.colors.LogNorm(vmin=1, vmax=1e4),
                               cmap='viridis', shading='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xlabel('AoI (degrees)')
            ax.set_ylabel(r'Wavelength ($\mu$m)')
            ax.set_title(f'R_bulk/R_GRIN \u2014 {thick_c:.1f} \u03bcm, {pol_c}-pol')
            _annotate_geomean(ax, ratio)
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_colorplot_ratio_bulk_over_GRIN.png', dpi=150)
    plt.close()
    print("  Saved fig_colorplot_ratio_bulk_over_GRIN.png")


def fig_task2_thickness_sweep(S):
    """Task 2: Thickness sweep for all loss shapes (4 parameter combos)."""
    print("\n=== TASK 2: THICKNESS SWEEP - ALL LOSS SHAPES ===")

    sweep_configs = [
        dict(mode='wavelength', angle=80, pol='s', wavelength=None,
             title_suffix=r's-pol, 80\u00b0, $\lambda$-averaged 2\u20135 $\mu$m',
             fname='s80_wlavg'),
        dict(mode='wavelength', angle=45, pol='p', wavelength=None,
             title_suffix=r'p-pol, 45\u00b0, $\lambda$-averaged 2\u20135 $\mu$m',
             fname='p45_wlavg'),
        dict(mode='angle', angle=None, pol='s', wavelength=3.0,
             title_suffix=r's-pol, $\lambda$=3 $\mu$m, angle-averaged',
             fname='s_lam3_angavg'),
        dict(mode='angle', angle=None, pol='p', wavelength=4.5,
             title_suffix=r'p-pol, $\lambda$=4.5 $\mu$m, angle-averaged',
             fname='p_lam4p5_angavg'),
    ]

    k_values_shapes = np.array([2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100])
    shape_names_t2 = ['sKK (HT deriv)', 'Constant', 'Gaussian', 'Double peaks', 'Random']
    shape_colors_t2 = [GREEN, RED, BLUE, PURPLE, ORANGE]
    shape_markers_t2 = ['s', 'o', '^', 'D', 'v']
    angle_avg_list = np.arange(0, 90, 1)

    for cfg in sweep_configs:
        print(f"\n  --- Thickness sweep: {cfg['fname']} ---")
        R_shapes_vs_k = {name: [] for name in shape_names_t2}
        thicknesses_t2 = []

        if cfg['mode'] == 'angle':
            idx_wl = np.argmin(np.abs(S.lamdata - cfg['wavelength']))
            n_sub_cfg = complex(S.ndata[idx_wl], S.kdata[idx_wl])

        for k_val in k_values_shapes:
            dx_k2 = 1 / (100 * k_val); xmin_k2 = -20 / k_val; xmax_k2 = -xmin_k2
            nx_k2 = 1 + int(np.floor((xmax_k2 - xmin_k2) / dx_k2))
            xx_k2 = np.linspace(xmin_k2, xmax_k2, nx_k2)
            e_re_k2 = tmm_h.logistic(xx_k2, k_val, S.nb)
            e_im_k2 = tmm_h.ht_derivative(xx_k2, e_re_k2)
            loss_ref_k2 = np.trapezoid(e_im_k2, xx_k2)
            thickness_k2 = xmax_k2 - xmin_k2
            thicknesses_t2.append(thickness_k2)

            shapes_k2 = [
                np.copy(e_im_k2),
                make_constant_profile(xx_k2, loss_ref_k2),
                make_gaussian_profile(xx_k2, loss_ref_k2),
                make_double_peak_profile(xx_k2, loss_ref_k2),
                make_random_profile(xx_k2, loss_ref_k2),
            ]

            for name, e_im_s in zip(shape_names_t2, shapes_k2):
                nc_s2, dc_s2 = tmm_h.discretize_profile(xx_k2, e_re_k2 + 1j * e_im_s, delta=S.delta)
                if cfg['mode'] == 'wavelength':
                    Rb_s2, _ = Rback_vs_wavelength(nc_s2, dc_s2, S.ndata, S.kdata, S.lamdata,
                                                    cfg['angle'], cfg['pol'])
                    R_avg = np.trapezoid(Rb_s2, S.lamdata) / (S.lamdata[-1] - S.lamdata[0])
                else:
                    Rb_s2, _ = Rback_vs_angle(nc_s2, dc_s2, n_sub_cfg,
                                               angle_avg_list, cfg['wavelength'], cfg['pol'])
                    R_avg = np.trapezoid(Rb_s2, angle_avg_list) / (angle_avg_list[-1] - angle_avg_list[0])
                R_shapes_vs_k[name].append(R_avg)

            print(f"    k={k_val:3d} ({thickness_k2:.1f} um): done")

        thicknesses_t2 = np.array(thicknesses_t2)

        if cfg['mode'] == 'wavelength':
            Rb_bulk_cfg, _ = Rback_bulk_wl(S.ndata, S.kdata, S.lamdata, cfg['angle'], cfg['pol'])
            Rb_avg_bulk_cfg = np.trapezoid(Rb_bulk_cfg, S.lamdata) / (S.lamdata[-1] - S.lamdata[0])
        else:
            Rb_bulk_cfg = Rback_bulk_angle(n_sub_cfg, angle_avg_list, cfg['wavelength'], cfg['pol'])
            Rb_avg_bulk_cfg = np.trapezoid(Rb_bulk_cfg, angle_avg_list) / (angle_avg_list[-1] - angle_avg_list[0])

        fig_t2, ax_t2 = plt.subplots(1, 1, figsize=(9, 5.5))
        ax_t2.axhline(Rb_avg_bulk_cfg, color='gray', lw=1.5, ls=':', label='Bulk sapphire', zorder=1)
        for name, col, mk in zip(shape_names_t2, shape_colors_t2, shape_markers_t2):
            ax_t2.plot(thicknesses_t2, R_shapes_vs_k[name], f'{mk}-', color=col,
                       lw=2, ms=6, label=name)
        ax_t2.set_xlabel(r'Coating thickness ($\mu$m)')
        ax_t2.set_ylabel(r'$\langle R_{\rm back} \rangle$')
        ax_t2.set_title(f'Thickness sweep \u2014 all loss shapes ({cfg["title_suffix"]})')
        ax_t2.set_xscale('log')
        ax_t2.set_yscale('log')
        ax_t2.legend(fontsize=10, loc='best')
        plt.tight_layout()
        plt.savefig(f'{S.FIGDIR}/fig_thickness_sweep_all_shapes_{cfg["fname"]}.png', dpi=150)
        plt.close()
        print(f"  Saved fig_thickness_sweep_all_shapes_{cfg['fname']}.png")


def fig_task3_losses_matched(S):
    """Task 3: Losses-matched thin vs thick comparison."""
    print("\n=== TASK 3: LOSSES-MATCHED THIN vs THICK ===")

    sweep_configs = [
        dict(mode='wavelength', angle=80, pol='s', wavelength=None,
             title_suffix=r's-pol, 80\u00b0, $\lambda$-averaged 2\u20135 $\mu$m',
             fname='s80_wlavg'),
        dict(mode='wavelength', angle=45, pol='p', wavelength=None,
             title_suffix=r'p-pol, 45\u00b0, $\lambda$-averaged 2\u20135 $\mu$m',
             fname='p45_wlavg'),
        dict(mode='angle', angle=None, pol='s', wavelength=3.0,
             title_suffix=r's-pol, $\lambda$=3 $\mu$m, angle-averaged',
             fname='s_lam3_angavg'),
        dict(mode='angle', angle=None, pol='p', wavelength=4.5,
             title_suffix=r'p-pol, $\lambda$=4.5 $\mu$m, angle-averaged',
             fname='p_lam4p5_angavg'),
    ]

    # Pre-compute coating stacks (same for all configs)
    loss_thin_t3 = np.trapezoid(S.e_im_deriv, S.xx)
    loss_thick_t3 = np.trapezoid(S.e_im_thick, S.xx_thick)
    scale_factor = loss_thick_t3 / loss_thin_t3
    e_im_matched_t3 = S.e_im_deriv * scale_factor
    nc_thin_t3, dc_thin_t3 = tmm_h.discretize_profile(S.xx, S.e_re + 1j * S.e_im_deriv, delta=S.delta)
    nc_thick_t3, dc_thick_t3 = tmm_h.discretize_profile(S.xx_thick, S.e_re_thick + 1j * S.e_im_thick, delta=S.delta)
    nc_matched_t3, dc_matched_t3 = tmm_h.discretize_profile(S.xx, S.e_re + 1j * e_im_matched_t3, delta=S.delta)
    thin_thick_um = S.xx[-1] - S.xx[0]
    thick_thick_um = S.xx_thick[-1] - S.xx_thick[0]
    print(f"  Thin ({thin_thick_um:.1f} um): int(eps'')dx = {loss_thin_t3:.6f}")
    print(f"  Thick ({thick_thick_um:.1f} um): int(eps'')dx = {loss_thick_t3:.6f}")
    print(f"  Loss ratio = {scale_factor:.3f}x")

    angle_avg_list_t3 = np.arange(0, 90, 1)

    for cfg in sweep_configs:
        print(f"\n  --- Losses-matched: {cfg['fname']} ---")

        if cfg['mode'] == 'wavelength':
            xaxis = S.lamdata
            xlabel = r'Wavelength ($\mu$m)'
            Rb_thin, A_thin = Rback_vs_wavelength(nc_thin_t3, dc_thin_t3, S.ndata, S.kdata,
                                                   S.lamdata, cfg['angle'], cfg['pol'])
            Rb_thick, A_thick = Rback_vs_wavelength(nc_thick_t3, dc_thick_t3, S.ndata, S.kdata,
                                                     S.lamdata, cfg['angle'], cfg['pol'])
            Rb_matched, A_matched = Rback_vs_wavelength(nc_matched_t3, dc_matched_t3, S.ndata, S.kdata,
                                                         S.lamdata, cfg['angle'], cfg['pol'])
            span = S.lamdata[-1] - S.lamdata[0]
            xint = S.lamdata
        else:
            xaxis = angle_avg_list_t3
            xlabel = 'Angle of Incidence (deg)'
            idx_wl = np.argmin(np.abs(S.lamdata - cfg['wavelength']))
            n_sub_t3 = complex(S.ndata[idx_wl], S.kdata[idx_wl])
            Rb_thin, A_thin = Rback_vs_angle(nc_thin_t3, dc_thin_t3, n_sub_t3,
                                              angle_avg_list_t3, cfg['wavelength'], cfg['pol'])
            Rb_thick, A_thick = Rback_vs_angle(nc_thick_t3, dc_thick_t3, n_sub_t3,
                                                angle_avg_list_t3, cfg['wavelength'], cfg['pol'])
            Rb_matched, A_matched = Rback_vs_angle(nc_matched_t3, dc_matched_t3, n_sub_t3,
                                                    angle_avg_list_t3, cfg['wavelength'], cfg['pol'])
            span = angle_avg_list_t3[-1] - angle_avg_list_t3[0]
            xint = angle_avg_list_t3

        Ravg_thin = np.trapezoid(Rb_thin, xint) / span
        Ravg_thick = np.trapezoid(Rb_thick, xint) / span
        Ravg_matched = np.trapezoid(Rb_matched, xint) / span
        Aavg_thin = np.trapezoid(A_thin, xint) / span
        Aavg_thick = np.trapezoid(A_thick, xint) / span
        Aavg_matched = np.trapezoid(A_matched, xint) / span
        print(f"  Thin:    <R>={Ravg_thin:.5f},  <A>={Aavg_thin:.5f}")
        print(f"  Thick:   <R>={Ravg_thick:.5f},  <A>={Aavg_thick:.5f}")
        print(f"  Matched: <R>={Ravg_matched:.5f},  <A>={Aavg_matched:.5f}")

        fig_t3, (ax_t3l, ax_t3r) = plt.subplots(1, 2, figsize=(13, 5))

        ax_t3l.plot(xaxis, Rb_thin, color=GREEN, lw=2.5,
                    label=rf'Thin ({thin_thick_um:.1f} \u03bcm), $\langle R \rangle$={Ravg_thin:.4f}')
        ax_t3l.plot(xaxis, Rb_thick, color=BLUE, lw=2.5,
                    label=rf'Thick ({thick_thick_um:.1f} \u03bcm), $\langle R \rangle$={Ravg_thick:.4f}')
        ax_t3l.plot(xaxis, Rb_matched, color=RED, lw=2.5, ls='--',
                    label=rf'Thin + {scale_factor:.1f}\u00d7\u03b5\u2033, $\langle R \rangle$={Ravg_matched:.4f}')
        ax_t3l.set_xlabel(xlabel)
        ax_t3l.set_ylabel(r'$R_{\rm back}$')
        ax_t3l.set_title(f'Backside reflection ({cfg["title_suffix"]})')
        ax_t3l.legend(fontsize=9, loc='lower left')

        ax_t3r.plot(xaxis, A_thin, color=GREEN, lw=2.5,
                    label=rf'Thin ({thin_thick_um:.1f} \u03bcm), $\langle A \rangle$={Aavg_thin:.4f}')
        ax_t3r.plot(xaxis, A_thick, color=BLUE, lw=2.5,
                    label=rf'Thick ({thick_thick_um:.1f} \u03bcm), $\langle A \rangle$={Aavg_thick:.4f}')
        ax_t3r.plot(xaxis, A_matched, color=RED, lw=2.5, ls='--',
                    label=rf'Thin + {scale_factor:.1f}\u00d7\u03b5\u2033, $\langle A \rangle$={Aavg_matched:.4f}')
        ax_t3r.set_xlabel(xlabel)
        ax_t3r.set_ylabel(r'Absorption')
        ax_t3r.set_title(f'Absorption ({cfg["title_suffix"]})')
        ax_t3r.legend(fontsize=9, loc='best')

        plt.tight_layout()
        plt.savefig(f'{S.FIGDIR}/fig_losses_matched_{cfg["fname"]}.png', dpi=150)
        plt.close()
        print(f"  Saved fig_losses_matched_{cfg['fname']}.png")


# ============================================================================
# Spectral FoM analysis figures (absorbed from standalone scripts)
# ============================================================================

def fig_crossover(S):
    """Spectral crossover: logistic (1/k² → exp) and Lorentzian (pure exp)."""
    print("\n=== SPECTRAL CROSSOVER ===")

    # ---- Lorentzian M=2000 grid ----
    a_l, gam_l, nb_l = 1.0, 0.01, 1.5
    M_l = 2000
    dx_l = gam_l / 100
    xmin_l = -M_l * gam_l
    xx_l = np.linspace(xmin_l, -xmin_l, 1 + int(np.floor(-2 * xmin_l / dx_l)))
    ee_l = tmm_h.eps(xx_l, a_l, gam_l, nb_l)
    u_l = np.real(ee_l)
    v_l = np.imag(ee_l)

    # ---- Figure 1: Logistic crossover ----
    k_num_g, pwr_num_g = _spectrum_physical(S.xx_fom, S.e_re_fom, S.e_im_fom)
    mask_g = k_num_g > 0.5
    k_pos_g = k_num_g[mask_g]
    pwr_pos_g = pwr_num_g[mask_g]

    k_an = np.linspace(0.5, 250, 2000)
    Delta = S.nb**2 - 1
    C_exact_g = 4 * np.pi**2 * Delta**2 / S.k_steep**2
    pwr_sinh = C_exact_g / np.sinh(np.pi * k_an / S.k_steep)**2
    pwr_1overk2 = 4 * Delta**2 / k_an**2
    pwr_exp_g = 4 * C_exact_g * np.exp(-2 * np.pi * k_an / S.k_steep)
    k_cross = S.k_steep / np.pi

    fig, (ax_prof, ax_spec) = plt.subplots(1, 2, figsize=(15, 7))
    zoom = (S.xx_fom >= -0.05) & (S.xx_fom <= 0.05)
    ax2 = ax_prof.twinx()
    ax_prof.plot(S.xx_fom[zoom], S.e_re_fom[zoom], color=BLUE, lw=2.0)
    ax2.plot(S.xx_fom[zoom], S.e_im_fom[zoom], color=RED, lw=2.0)
    ax_prof.set_xlabel(r'$x$ ($\mu$m)')
    ax_prof.set_ylabel(r"$\varepsilon'(x)$", color=BLUE)
    ax2.set_ylabel(r"$\varepsilon''(x)$", color=RED)
    ax_prof.tick_params(axis='y', labelcolor=BLUE)
    ax2.tick_params(axis='y', labelcolor=RED)
    ax_prof.set_title(r'Logistic profile ($k_s=100\ \mu$m$^{-1}$, $n_b=1.7$)')
    ax_prof.set_xlim(-0.05, 0.05)

    ax_spec.semilogy(k_pos_g, pwr_pos_g, '-', color=GREEN, lw=2.0,
                     label=f'Numerical (M={2000})')
    ax_spec.semilogy(k_an, pwr_sinh, '--', color='black', lw=1.5,
                     label=r'$C/\sinh^2(\pi k / k_s)$')
    ax_spec.semilogy(k_an, pwr_1overk2, ':', color='#d62728', lw=2.0,
                     label=r'$1/k^2$ (small $k$)')
    ax_spec.semilogy(k_an, pwr_exp_g, ':', color=BLUE, lw=2.0,
                     label=r'$\exp(-2\pi k/k_s)$ (large $k$)')
    ax_spec.axvline(k_cross, color='gray', ls='--', lw=1.0, alpha=0.7)
    ax_spec.text(k_cross + 3, 1e-7, f'$k = k_s/\\pi \\approx {k_cross:.0f}$',
                 fontsize=11, color='gray')
    ax_spec.set_xlabel(r'Spatial frequency $k$ ($\mu$m$^{-1}$)')
    ax_spec.set_ylabel(r'$|\hat{\varepsilon}(k)|^2$')
    ax_spec.set_title(r'Logistic spectrum: $1/k^2 \to \exp(-2\pi k/k_s)$ crossover at $k \approx k_s/\pi$')
    ax_spec.set_xlim(0.5, 250)
    ax_spec.set_ylim(1e-9, 1e3)
    ax_spec.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_logistic_sinh_crossover_unnorm.png')
    plt.close()
    print("  Saved fig_logistic_sinh_crossover_unnorm.png")

    # ---- Figure 2: Lorentzian crossover ----
    k_num_l, pwr_num_l = _spectrum_physical(xx_l, u_l, v_l)
    mask_l = k_num_l > 0.5
    k_pos_l = k_num_l[mask_l]
    pwr_pos_l = pwr_num_l[mask_l]

    C_exact_l = 4 * np.pi**2 * a_l**2 * gam_l**2
    pwr_exp_an = C_exact_l * np.exp(-2 * gam_l * k_an)

    fig, (ax_prof, ax_spec) = plt.subplots(1, 2, figsize=(15, 7))
    zoom_l = (xx_l >= -0.05) & (xx_l <= 0.05)
    ax2 = ax_prof.twinx()
    ax_prof.plot(xx_l[zoom_l], u_l[zoom_l], color=BLUE, lw=2.0)
    ax2.plot(xx_l[zoom_l], v_l[zoom_l], color=RED, lw=2.0)
    ax_prof.set_xlabel(r'$x$ ($\mu$m)')
    ax_prof.set_ylabel(r"$\varepsilon'(x)$", color=BLUE)
    ax2.set_ylabel(r"$\varepsilon''(x)$", color=RED)
    ax_prof.tick_params(axis='y', labelcolor=BLUE)
    ax2.tick_params(axis='y', labelcolor=RED)
    ax_prof.set_title(r'Lorentzian profile ($\gamma=0.01\ \mu$m, $n_b=1.5$, $a=1$)')
    ax_prof.set_xlim(-0.05, 0.05)

    ax_spec.semilogy(k_pos_l, pwr_pos_l, '-', color=GREEN, lw=2.0,
                     label=f'Numerical (M={M_l})')
    ax_spec.semilogy(k_an, pwr_exp_an, '--', color='black', lw=1.8,
                     label=r'$4\pi^2 a^2\gamma^2\,\exp(-2\gamma k)$  [exact]')
    ax_spec.set_xlabel(r'Spatial frequency $k$ ($\mu$m$^{-1}$)')
    ax_spec.set_ylabel(r'$|\hat{\varepsilon}(k)|^2$')
    ax_spec.set_title(r'Lorentzian spectrum: pure $\exp(-2\gamma k)$, no crossover'
                      '\n' + r'(pole at $x=-i\gamma$ in lower half-plane)')
    ax_spec.set_xlim(0.5, 250)
    ax_spec.set_ylim(1e-6, 1e-2)
    ax_spec.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_lorentzian_crossover_M2000_unnorm.png')
    plt.close()
    print("  Saved fig_lorentzian_crossover_M2000_unnorm.png")


def fig_fom_spectrum(S):
    """Spectral FoM (green/red fill): logistic M=2000 and Lorentzian M=2000."""
    print("\n=== SPECTRAL FOM SPECTRUM ===")

    KLIM = 400

    # ---- Lorentzian M=2000 grid ----
    a_l, gam_l, nb_l = 1.0, 0.01, 1.5
    M_l = 2000
    dx_l = gam_l / 100
    xmin_l = -M_l * gam_l
    xx_l = np.linspace(xmin_l, -xmin_l, 1 + int(np.floor(-2 * xmin_l / dx_l)))
    ee_l = tmm_h.eps(xx_l, a_l, gam_l, nb_l)
    u_l = np.real(ee_l)
    v_l = np.imag(ee_l)

    # ---- Figure 1: Logistic FoM spectrum ----
    fom_g, k_g, pwr_g = tmm_h.skk_spectral_fom(S.xx_fom, S.e_re_fom, S.e_im_fom)
    print(f"  Logistic (M=2000): FoM = {fom_g:.2f}%")

    fig, (ax_prof, ax_fom) = plt.subplots(1, 2, figsize=(15, 7))
    zoom = (S.xx_fom >= -0.05) & (S.xx_fom <= 0.05)
    ax2 = ax_prof.twinx()
    ax_prof.plot(S.xx_fom[zoom], S.e_re_fom[zoom], color=BLUE, lw=2.0)
    ax2.plot(S.xx_fom[zoom], S.e_im_fom[zoom], color=RED, lw=2.0)
    ax_prof.set_xlabel(r'$x$ ($\mu$m)')
    ax_prof.set_ylabel(r"$\varepsilon'(x)$", color=BLUE)
    ax2.set_ylabel(r"$\varepsilon''(x)$", color=RED)
    ax_prof.tick_params(axis='y', labelcolor=BLUE)
    ax2.tick_params(axis='y', labelcolor=RED)
    ax_prof.set_title(r'Logistic profile ($k_s=100\ \mu$m$^{-1}$, $n_b=1.7$)')
    ax_prof.set_xlim(-0.05, 0.05)
    tmm_h.plot_spectral_fom(ax_fom, k_g, pwr_g, fom_g, klim=KLIM,
                            title=(r'Logistic ($M=2000$) — $d\varepsilon/dx \to$ FT $\to \div\, ik$'
                                   + f'\nFoM = {fom_g:.2f}%'))
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_logistic_fom_M2000_unnorm.png')
    plt.close()
    print("  Saved fig_logistic_fom_M2000_unnorm.png")

    # ---- Figure 2: Lorentzian FoM spectrum ----
    fom_l, k_l, pwr_l = tmm_h.skk_spectral_fom(xx_l, u_l, v_l)
    print(f"  Lorentzian (gam={gam_l}, M={M_l}): FoM = {fom_l:.2f}%")

    fig, (ax_prof, ax_fom) = plt.subplots(1, 2, figsize=(15, 7))
    zoom_l = (xx_l >= -0.05) & (xx_l <= 0.05)
    ax2 = ax_prof.twinx()
    ax_prof.plot(xx_l[zoom_l], u_l[zoom_l], color=BLUE, lw=2.0)
    ax2.plot(xx_l[zoom_l], v_l[zoom_l], color=RED, lw=2.0)
    ax_prof.set_xlabel(r'$x$ ($\mu$m)')
    ax_prof.set_ylabel(r"$\varepsilon'(x)$", color=BLUE)
    ax2.set_ylabel(r"$\varepsilon''(x)$", color=RED)
    ax_prof.tick_params(axis='y', labelcolor=BLUE)
    ax2.tick_params(axis='y', labelcolor=RED)
    ax_prof.set_title(r'Lorentzian profile ($\gamma=0.01\ \mu$m, $n_b=1.5$, $a=1$)')
    ax_prof.set_xlim(-0.05, 0.05)
    tmm_h.plot_spectral_fom(ax_fom, k_l, pwr_l, fom_l, klim=KLIM,
                            title=(r'Lorentzian ($\gamma=0.01\ \mu$m, $M=2000$) — $d\varepsilon/dx \to$ FT $\to \div\, ik$'
                                   + f'\nFoM = {fom_l:.2f}%'))
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_lorentzian_fom_M2000_unnorm.png')
    plt.close()
    print("  Saved fig_lorentzian_fom_M2000_unnorm.png")


def fig_fom_method(S):
    """2x2 comparison: direct FT vs derivative→FT→÷ik for Lorentzian and logistic."""
    print("\n=== FOM METHOD COMPARISON ===")

    KLIM = 400

    # ---- Lorentzian M=200 (gam=0.01, domain ±2 um) ----
    a_l, gam_l, nb_l = 1.0, 0.01, 1.5
    dx_l = gam_l / 100
    xmin_l = -gam_l * 200
    xx_l = np.linspace(xmin_l, -xmin_l, 1 + int(np.floor(-2 * xmin_l / dx_l)))
    ee_l = tmm_h.eps(xx_l, a_l, gam_l, nb_l)
    u_l = np.real(ee_l)
    v_l = np.imag(ee_l)

    # ---- Logistic M=20 (k_steep=100, domain ±0.2 um) ----
    k_steep_m = 100; nb_g = 1.7
    dx_g = 1 / (100 * k_steep_m)
    xmin_g = -20 / k_steep_m
    xx_g = np.linspace(xmin_g, -xmin_g, 1 + int(np.floor(-2 * xmin_g / dx_g)))
    u_g = (nb_g**2 - 1) / (1 + np.exp(k_steep_m * xx_g)) + 1
    v_g = tmm_h.ht_derivative(xx_g, u_g)

    # ---- Compute all 4 cases ----
    fom_a, k_a, pwr_a = _direct_ft_fom(xx_l, u_l, v_l)
    fom_b, k_b, pwr_b = tmm_h.skk_spectral_fom(xx_l, u_l, v_l)
    fom_c, k_c, pwr_c = _direct_ft_fom(xx_g, u_g, v_g)
    fom_d, k_d, pwr_d = tmm_h.skk_spectral_fom(xx_g, u_g, v_g)

    print(f"  (a) Lorentzian direct FT:          FoM = {fom_a:.2f}%")
    print(f"  (b) Lorentzian deriv-FT-integrate: FoM = {fom_b:.2f}%")
    print(f"  (c) Logistic direct FT:            FoM = {fom_c:.2f}%")
    print(f"  (d) Logistic deriv-FT-integrate:   FoM = {fom_d:.2f}%")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    configs = [
        (axes[0, 0], k_a, pwr_a, fom_a,
         r'(a) Lorentzian ($\gamma=0.01\ \mu$m) — direct FT of $\varepsilon(x)$'),
        (axes[0, 1], k_b, pwr_b, fom_b,
         r'(b) Lorentzian ($\gamma=0.01\ \mu$m) — $d\varepsilon/dx \to$ FT $\to \div\, ik$'),
        (axes[1, 0], k_c, pwr_c, fom_c,
         r'(c) Logistic — direct FT of $\varepsilon(x)$'),
        (axes[1, 1], k_d, pwr_d, fom_d,
         r'(d) Logistic — $d\varepsilon/dx \to$ FT $\to \div\, ik$'),
    ]
    for ax, k, pwr, fom, title in configs:
        tmm_h.plot_spectral_fom(ax, k, pwr, fom, klim=KLIM,
                                title=f'{title}\nFoM = {fom:.2f}%')
        ax.legend(loc='upper right', fontsize=9)

    fig.text(0.28, 0.98, r'Direct FT of $\varepsilon(x)$',
             fontsize=14, ha='center', va='top', fontweight='bold')
    fig.text(0.74, 0.98, r'Derivative $\to$ FT $\to$ integrate',
             fontsize=14, ha='center', va='top', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{S.FIGDIR}/fig_fom_integrate_comparison_narrow_unnorm.png')
    plt.close()
    print("  Saved fig_fom_integrate_comparison_narrow_unnorm.png")


def fig_profiles(S):
    """Dielectric profile gallery: 4 full-domain dual-axis plots."""
    print("\n=== DIELECTRIC PROFILE GALLERY ===")

    a_l, gam_l, nb_l = 1.0, 0.01, 1.5
    k_steep_p = 100; nb_g = 1.7
    dx_l = gam_l / 100
    dx_g = 1 / (100 * k_steep_p)

    def _save_profile(xx, u, v, title, fname):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax2 = ax.twinx()
        ax.plot(xx, u, color=BLUE, lw=2.0)
        ax2.plot(xx, v, color=RED, lw=2.0)
        ax.set_xlabel(r'$x$ ($\mu$m)')
        ax.set_ylabel(r"$\varepsilon'(x)$", color=BLUE)
        ax2.set_ylabel(r"$\varepsilon''(x)$", color=RED)
        ax.tick_params(axis='y', labelcolor=BLUE)
        ax2.tick_params(axis='y', labelcolor=RED)
        ax.set_title(title)
        ax.set_xlim(xx[0], xx[-1])
        plt.tight_layout()
        plt.savefig(f'{S.FIGDIR}/{fname}')
        plt.close()
        print(f"  Saved {fname}")

    # 1. Lorentzian M=200 (domain ±2 um)
    xmin_l200 = -gam_l * 200
    xx_l200 = np.linspace(xmin_l200, -xmin_l200, 1 + int(np.floor(-2 * xmin_l200 / dx_l)))
    ee_l200 = tmm_h.eps(xx_l200, a_l, gam_l, nb_l)
    _save_profile(xx_l200, np.real(ee_l200), np.imag(ee_l200),
                  r'Lorentzian profile ($\gamma=0.01\ \mu$m, $n_b=1.5$, $M=200$, domain $\pm 2\ \mu$m)',
                  'fig_profile_lorentzian_M200.png')

    # 2. Logistic M=20 (domain ±0.20 um)
    xmin_g20 = -20 / k_steep_p
    xx_g20 = np.linspace(xmin_g20, -xmin_g20, 1 + int(np.floor(-2 * xmin_g20 / dx_g)))
    u_g20 = (nb_g**2 - 1) / (1 + np.exp(k_steep_p * xx_g20)) + 1
    v_g20 = tmm_h.ht_derivative(xx_g20, u_g20)
    _save_profile(xx_g20, u_g20, v_g20,
                  r'Logistic profile ($k_s=100\ \mu$m$^{-1}$, $n_b=1.7$, $M=20$, domain $\pm 0.20\ \mu$m)',
                  'fig_profile_logistic_M20.png')

    # 3. Logistic M=2000 (domain ±20 um) — reuse S.xx_fom grid
    _save_profile(S.xx_fom, S.e_re_fom, S.e_im_fom,
                  r'Logistic profile ($k_s=100\ \mu$m$^{-1}$, $n_b=1.7$, $M=2000$, domain $\pm 20\ \mu$m)',
                  'fig_profile_logistic_M2000.png')

    # 4. Lorentzian M=2000 (domain ±20 um)
    xmin_l2000 = -2000 * gam_l
    xx_l2000 = np.linspace(xmin_l2000, -xmin_l2000, 1 + int(np.floor(-2 * xmin_l2000 / dx_l)))
    ee_l2000 = tmm_h.eps(xx_l2000, a_l, gam_l, nb_l)
    _save_profile(xx_l2000, np.real(ee_l2000), np.imag(ee_l2000),
                  r'Lorentzian profile ($\gamma=0.01\ \mu$m, $n_b=1.5$, $M=2000$, domain $\pm 20\ \mu$m)',
                  'fig_profile_lorentzian_M2000.png')


def fig_thick_colorplots(S):
    """Thick coating colorplots (50 µm and 100 µm): GRIN/sKK and bulk/GRIN ratios."""
    print("\n=== THICK COATING COLORPLOTS (50/100 um) ===")

    color_angle_list = np.arange(0, 90, 1)
    color_pols = ['s', 'p']
    k_vals = [0.8, 0.4]   # k=0.8 → 50 µm,  k=0.4 → 100 µm

    # ---- Bulk (coating-independent) ----
    Rb_bulk_2D_d = {}
    for pol_c in color_pols:
        print(f"  Computing bulk 2D ({pol_c}-pol)...")
        Rb_bulk_2D_d[pol_c] = Rback_bulk_2D(S.ndata, S.kdata, S.lamdata,
                                              color_angle_list, pol_c)

    # ---- sKK and GRIN for each thickness ----
    Rb_skk_2D = {pol_c: [] for pol_c in color_pols}
    Rb_grin_2D = {pol_c: [] for pol_c in color_pols}
    thicknesses = []

    for k_c in k_vals:
        dx_c = 1 / (100 * k_c)
        xmin_c = -20 / k_c
        nx_c = 1 + int(np.floor(-2 * xmin_c / dx_c))
        xx_c = np.linspace(xmin_c, -xmin_c, nx_c)
        e_re_c = tmm_h.logistic(xx_c, k_c, S.nb)
        e_im_c = tmm_h.ht_derivative(xx_c, e_re_c)
        thicknesses.append(-2 * xmin_c)

        nc_skk_c, dc_skk_c = tmm_h.discretize_profile(xx_c, e_re_c + 1j * e_im_c,
                                                        delta=S.delta)
        nc_grin_c, dc_grin_c = tmm_h.discretize_profile(xx_c, e_re_c + 0j,
                                                          delta=S.delta)
        for pol_c in color_pols:
            print(f"  k={k_c} ({-2*xmin_c:.0f} µm), {pol_c}-pol: sKK 2D...")
            Rb_s, _ = Rback_2D(nc_skk_c, dc_skk_c, S.ndata, S.kdata, S.lamdata,
                               color_angle_list, pol_c)
            Rb_skk_2D[pol_c].append(Rb_s)
            print(f"  k={k_c} ({-2*xmin_c:.0f} µm), {pol_c}-pol: GRIN 2D...")
            Rb_g, _ = Rback_2D(nc_grin_c, dc_grin_c, S.ndata, S.kdata, S.lamdata,
                               color_angle_list, pol_c)
            Rb_grin_2D[pol_c].append(Rb_g)

    # ---- Figure 1: GRIN/sKK ratio (2×2) ----
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))
    for col, (k_c, thick_c) in enumerate(zip(k_vals, thicknesses)):
        for row, pol_c in enumerate(color_pols):
            ax = axes1[row, col]
            ratio = Rb_grin_2D[pol_c][col] / np.clip(Rb_skk_2D[pol_c][col], 1e-10, None)
            im = ax.pcolormesh(color_angle_list, S.lamdata, ratio.T,
                               norm=matplotlib.colors.LogNorm(vmin=1, vmax=1e4),
                               cmap='viridis', shading='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xlabel('AoI (degrees)')
            ax.set_ylabel(r'Wavelength ($\mu$m)')
            ax.set_title(f'R_GRIN/R_sKK — {thick_c:.0f} µm, {pol_c}-pol')
            _annotate_geomean(ax, ratio)
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_colorplot_ratio_GRIN_over_sKK_thick.png', dpi=150)
    plt.close()
    print("  Saved fig_colorplot_ratio_GRIN_over_sKK_thick.png")

    # ---- Figure 2: bulk/GRIN ratio (2×2) ----
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    for col, (k_c, thick_c) in enumerate(zip(k_vals, thicknesses)):
        for row, pol_c in enumerate(color_pols):
            ax = axes2[row, col]
            ratio = Rb_bulk_2D_d[pol_c] / np.clip(Rb_grin_2D[pol_c][col], 1e-10, None)
            im = ax.pcolormesh(color_angle_list, S.lamdata, ratio.T,
                               norm=matplotlib.colors.LogNorm(vmin=1, vmax=1e4),
                               cmap='viridis', shading='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xlabel('AoI (degrees)')
            ax.set_ylabel(r'Wavelength ($\mu$m)')
            ax.set_title(f'R_bulk/R_GRIN — {thick_c:.0f} µm, {pol_c}-pol')
            _annotate_geomean(ax, ratio)
    plt.tight_layout()
    plt.savefig(f'{S.FIGDIR}/fig_colorplot_ratio_bulk_over_GRIN_thick.png', dpi=150)
    plt.close()
    print("  Saved fig_colorplot_ratio_bulk_over_GRIN_thick.png")


# ============================================================================
# Figure registry
# ============================================================================
FIGURE_MAP = {
    'fig1':  ('Lorentzian vs GRIN',           fig_lorentz_vs_grin),
    'fig2':  ('Endpoint problem',             fig_endpoint_problem),
    'fig3':  ('Derivative result',            fig_derivative_result),
    'fig4':  ('Backside reflection',          fig_reflection_wavelength),
    'fig5':  ('FoM intro',                    fig_fom_intro),
    'fig6':  ('Alpha tradeoff',              fig_alpha_tradeoff),
    'fig7':  ('Sigma gating',                fig_sigma_gating),
    'fig8':  ('FoM comparison',              fig_fom_comparison),
    'fig9':  ('Angle-resolved',              fig_angle_resolved),
    'fig10': ('Thickness design space',      fig_thickness_single),
    'loss_shapes':     ('Loss shape comparison',      fig_loss_shapes),
    'thick_shapes':    ('Thick coating shapes',       fig_thick_shapes),
    'task1': ('2D colorplots',               fig_task1_colorplots),
    'task2': ('Thickness sweep all shapes',  fig_task2_thickness_sweep),
    'task3': ('Losses-matched comparison',   fig_task3_losses_matched),
    'crossover':      ('Spectral crossover (logistic + Lorentzian)',  fig_crossover),
    'fom_spectrum':   ('Spectral FoM plots (logistic + Lorentzian)',  fig_fom_spectrum),
    'fom_method':     ('Direct FT vs derivative method comparison',   fig_fom_method),
    'profiles':       ('Dielectric profile gallery (4 panels)',       fig_profiles),
    'thick_colorplots': ('Thick coating colorplots (50/100 um)',      fig_thick_colorplots),
}


# ============================================================================
# CLI entry point
# ============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='sKK Analysis — generate paper figures selectively.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('figures', nargs='*', default=['all'],
                        help='Figure names to generate (default: all). Use --list to see options.')
    parser.add_argument('--list', action='store_true',
                        help='List available figure names and exit.')
    parser.add_argument('--outdir', default=None,
                        help='Output directory for figures (default: theory/figures/).')
    args = parser.parse_args()

    if args.list:
        print("Available figures:")
        for key, (desc, _) in FIGURE_MAP.items():
            print(f"  {key:<20s} {desc}")
        sys.exit(0)

    S = setup(figdir=args.outdir)

    if 'all' in args.figures:
        targets = list(FIGURE_MAP.keys())
    else:
        targets = args.figures

    for name in targets:
        if name not in FIGURE_MAP:
            print(f"Unknown figure: '{name}'. Use --list to see options.")
            continue
        desc, fn = FIGURE_MAP[name]
        print(f"\n{'='*60}")
        print(f"  {name}: {desc}")
        print(f"{'='*60}")
        fn(S)

    print("\n=== ALL REQUESTED FIGURES GENERATED ===")
