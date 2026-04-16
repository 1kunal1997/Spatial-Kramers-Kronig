"""
Constant-loss different-shape explorer
======================================
Thickness sweeps of reflectance for different ε'' shapes (sKK, constant,
Gaussian, double-peak, random) with matched total losses.

Geometry:  n_b (semi-inf) | coating | air (semi-inf)
No substrate — directly measures the coating's AR performance.

Three modes:
  sweep_wavelength_averaged(angle, pol, ...)  — ⟨R⟩ over wavelengths
  sweep_angle_averaged(wavelength, pol, ...)  — ⟨R⟩ over angles
  sweep_single(wavelength, angle, pol, ...)   — R at one (λ, θ) point

Usage:
  python theory/constant_loss_different_shape.py          # default 4 configs
  python theory/constant_loss_different_shape.py --help   # see all options
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tmm
import tmm_helper as tmm_h
from scipy.ndimage import gaussian_filter1d

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

SHAPE_NAMES = ['sKK (HT deriv)', 'Constant', 'Gaussian', 'Double peaks', 'Random']
SHAPE_COLORS = [GREEN, RED, BLUE, PURPLE, ORANGE]
SHAPE_MARKERS = ['s', 'o', '^', 'D', 'v']

# ============================================================================
# Loss shape generators
# ============================================================================
def make_constant_profile(xx, target_integral):
    L = xx[-1] - xx[0]
    c = target_integral / L
    return np.full_like(xx, max(c, 0.0))

def make_gaussian_profile(xx, target_integral):
    sigma_g = (xx[-1] - xx[0]) / 8
    g = np.exp(-xx**2 / (2 * sigma_g**2))
    integral = np.trapezoid(g, xx)
    if integral > 0:
        g *= target_integral / integral
    return np.maximum(g, 0.0)

def make_double_peak_profile(xx, target_integral):
    L = xx[-1] - xx[0]
    sigma_g = L / 16
    x1, x2 = -L/4, L/4
    g = np.exp(-(xx - x1)**2 / (2*sigma_g**2)) + np.exp(-(xx - x2)**2 / (2*sigma_g**2))
    integral = np.trapezoid(g, xx)
    if integral > 0:
        g *= target_integral / integral
    return np.maximum(g, 0.0)

def make_random_profile(xx, target_integral, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(xx))
    sigma_filt = len(xx) // 20
    smooth = gaussian_filter1d(noise, sigma_filt)
    smooth -= smooth.min()
    integral = np.trapezoid(smooth, xx)
    if integral > 0:
        smooth *= target_integral / integral
    return smooth

# ============================================================================
# TMM helpers — substrate-free geometry: n_b | coating | air
# ============================================================================
def R_vs_wavelength(n_coating, d_coating, nb, lamdata, angle_air_deg, pol):
    """Reflectance vs wavelength.  Stack: n_b (inf) | coating | air (inf).

    angle_air_deg is the angle in air; internally converted to angle in n_b.
    """
    angle_nb = angle_air_to_nb(angle_air_deg, nb)
    n_t = [nb] + list(n_coating) + [1]
    d_t = [np.inf] + list(d_coating) + [np.inf]
    R_arr = np.zeros(len(lamdata))
    for i, wl in enumerate(lamdata):
        T, R, A = tmm_h.TRA(n_t, d_t, lamb=wl, angle=angle_nb, pol=pol)
        R_arr[i] = R
    return R_arr

def R_vs_angle(n_coating, d_coating, nb, angle_list_air_deg, lam, pol):
    """Reflectance vs angle at a single wavelength.

    angle_list_air_deg are angles in air; internally converted to angles in n_b.
    """
    n_t = [nb] + list(n_coating) + [1]
    d_t = [np.inf] + list(d_coating) + [np.inf]
    R_arr = np.zeros(len(angle_list_air_deg))
    for i, ang in enumerate(angle_list_air_deg):
        angle_nb = angle_air_to_nb(ang, nb)
        T, R, A = tmm_h.TRA(n_t, d_t, lamb=lam, angle=angle_nb, pol=pol)
        R_arr[i] = R
    return R_arr

def R_at_point(n_coating, d_coating, nb, lam, angle_air_deg, pol):
    """Reflectance at a single (λ, θ) point.

    angle_air_deg is the angle in air; internally converted to angle in n_b.
    """
    angle_nb = angle_air_to_nb(angle_air_deg, nb)
    n_t = [nb] + list(n_coating) + [1]
    d_t = [np.inf] + list(d_coating) + [np.inf]
    T, R, A = tmm_h.TRA(n_t, d_t, lamb=lam, angle=angle_nb, pol=pol)
    return R

def angle_air_to_nb(angle_air_deg, nb):
    """Convert angle in air to angle in n_b via Snell's law."""
    return np.arcsin(np.sin(np.radians(angle_air_deg)) / nb)

def R_bare_interface(nb, angle_air_deg, pol):
    """Fresnel reflectance at a bare n_b | air interface (the no-coating limit).

    angle_air_deg is the angle in air; internally converted to angle in n_b.
    """
    angle_nb = angle_air_to_nb(angle_air_deg, nb)
    th_f = tmm.snell(nb, 1, angle_nb)
    return tmm.interface_R(pol, nb, 1, angle_nb, th_f)

# ============================================================================
# Profile builder
# ============================================================================
def build_profiles(k_steep, nb):
    """Build ε' and all ε'' shapes for a given logistic steepness.

    Returns (xx, e_re, shapes_dict, thickness).
    """
    dx = 1 / (100 * k_steep)
    xmin = -20 / k_steep
    nx = 1 + int(np.floor(-2 * xmin / dx))
    xx = np.linspace(xmin, -xmin, nx)
    e_re = tmm_h.logistic(xx, k_steep, nb)
    e_im_skk = tmm_h.ht_derivative(xx, e_re)
    loss_ref = np.trapezoid(e_im_skk, xx)
    thickness = -2 * xmin

    shapes = {
        'sKK (HT deriv)': e_im_skk,
        'Constant': make_constant_profile(xx, loss_ref),
        'Gaussian': make_gaussian_profile(xx, loss_ref),
        'Double peaks': make_double_peak_profile(xx, loss_ref),
        'Random': make_random_profile(xx, loss_ref),
    }
    return xx, e_re, shapes, thickness

# ============================================================================
# Core sweep functions
# ============================================================================
K_VALUES_DEFAULT = np.array([2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100])
NB = 1.7
DELTA = 0.01

# Grid sweep defaults
THICKNESS_GRID = np.geomspace(0.1, 100.0, 31)   # 10 pts/decade, 0.1–100 µm
LAM_GRID = np.unique(np.concatenate([
    np.round(np.arange(0.5, 1.05, 0.1), 10),    # 0.5, 0.6, …, 1.0  (6 pts)
    np.round(np.arange(1.0, 10.5,  1.0), 10),   # 1.0, 2.0, …, 10.0 (10 pts)
]))                                               # 15 unique wavelengths
ANGLES_GRID = np.array([0.0, 30.0, 45.0, 60.0, 80.0])   # degrees in air

def sweep_wavelength_averaged(angle_deg, pol, lam_min=2.0, lam_max=5.0, n_lam=300,
                               k_values=None, nb=NB, delta=DELTA,
                               figdir=None, show=False):
    """Thickness sweep with R averaged over wavelengths.

    Parameters
    ----------
    angle_deg : float  — incidence angle in degrees
    pol : str          — 's' or 'p'
    lam_min, lam_max : float — wavelength range (µm)
    n_lam : int        — number of wavelength points
    """
    if k_values is None:
        k_values = K_VALUES_DEFAULT
    lamdata = np.linspace(lam_min, lam_max, n_lam)
    figdir = figdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(figdir, exist_ok=True)

    R_shapes = {name: [] for name in SHAPE_NAMES}
    thicknesses = []

    for k_val in k_values:
        xx, e_re, shapes, thick = build_profiles(k_val, nb)
        thicknesses.append(thick)
        for name in SHAPE_NAMES:
            nc, dc = tmm_h.discretize_profile(xx, e_re + 1j * shapes[name], delta=delta)
            R_arr = R_vs_wavelength(nc, dc, nb, lamdata, angle_deg, pol)
            R_avg = np.trapezoid(R_arr, lamdata) / (lamdata[-1] - lamdata[0])
            R_shapes[name].append(R_avg)
        print(f"  k={k_val:.4g} ({thick:.1f} µm): done")

    thicknesses = np.array(thicknesses)

    # Bare interface reference
    R_bare = R_bare_interface(nb, angle_deg, pol)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.axhline(R_bare, color='gray', lw=1.5, ls=':', label='Bare interface', zorder=1)
    for name, col, mk in zip(SHAPE_NAMES, SHAPE_COLORS, SHAPE_MARKERS):
        ax.plot(thicknesses, R_shapes[name], f'{mk}-', color=col, lw=2, ms=6, label=name)
    ax.set_xlabel(r'Coating thickness ($\mu$m)')
    ax.set_ylabel(r'$\langle R \rangle$')
    ax.set_title(f'Thickness sweep — {pol}-pol, {angle_deg}°, '
                 f'λ-avg {lam_min}–{lam_max} µm')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    fname = f'explore_thickness_wlavg_{pol}{int(angle_deg)}_{lam_min}-{lam_max}um.png'
    plt.savefig(os.path.join(figdir, fname), dpi=150)
    print(f"  Saved {fname}")
    if show:
        plt.show()
    else:
        plt.close()

    return thicknesses, R_shapes, R_bare


def sweep_angle_averaged(wavelength, pol, angle_min=0, angle_max=89,
                          k_values=None, nb=NB, delta=DELTA,
                          figdir=None, show=False):
    """Thickness sweep with R averaged over angles.

    Parameters
    ----------
    wavelength : float — wavelength in µm
    pol : str          — 's' or 'p'
    angle_min, angle_max : float — angle range (degrees)
    """
    if k_values is None:
        k_values = K_VALUES_DEFAULT
    angle_list = np.arange(angle_min, angle_max + 1, 1)
    figdir = figdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(figdir, exist_ok=True)

    R_shapes = {name: [] for name in SHAPE_NAMES}
    thicknesses = []

    for k_val in k_values:
        xx, e_re, shapes, thick = build_profiles(k_val, nb)
        thicknesses.append(thick)
        for name in SHAPE_NAMES:
            nc, dc = tmm_h.discretize_profile(xx, e_re + 1j * shapes[name], delta=delta)
            R_arr = R_vs_angle(nc, dc, nb, angle_list, wavelength, pol)
            R_avg = np.trapezoid(R_arr, angle_list) / (angle_list[-1] - angle_list[0])
            R_shapes[name].append(R_avg)
        print(f"  k={k_val:.4g} ({thick:.1f} µm): done")

    thicknesses = np.array(thicknesses)

    # Bare interface reference (angle-averaged)
    R_bare_arr = np.array([R_bare_interface(nb, a, pol) for a in angle_list])
    R_bare_avg = np.trapezoid(R_bare_arr, angle_list) / (angle_list[-1] - angle_list[0])

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.axhline(R_bare_avg, color='gray', lw=1.5, ls=':', label='Bare interface', zorder=1)
    for name, col, mk in zip(SHAPE_NAMES, SHAPE_COLORS, SHAPE_MARKERS):
        ax.plot(thicknesses, R_shapes[name], f'{mk}-', color=col, lw=2, ms=6, label=name)
    ax.set_xlabel(r'Coating thickness ($\mu$m)')
    ax.set_ylabel(r'$\langle R \rangle$')
    ax.set_title(f'Thickness sweep — {pol}-pol, λ={wavelength} µm, '
                 f'angle-avg {angle_min}–{angle_max}°')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    fname = f'explore_thickness_angavg_{pol}_lam{wavelength}um.png'
    plt.savefig(os.path.join(figdir, fname), dpi=150)
    print(f"  Saved {fname}")
    if show:
        plt.show()
    else:
        plt.close()

    return thicknesses, R_shapes, R_bare_avg


def sweep_single(wavelength, angle_deg, pol,
                  thickness_list=None, nb=NB, delta=DELTA,
                  figdir=None, show=False):
    """Thickness sweep at a single (λ, θ) — no averaging.

    Parameters
    ----------
    wavelength     : float — wavelength in µm
    angle_deg      : float — incidence angle in degrees (in air)
    pol            : str   — 's' or 'p'
    thickness_list : array — coating thicknesses in µm (default: geomspace 0.1–100)
    """
    if thickness_list is None:
        thickness_list = np.geomspace(0.1, 100.0, 31)
    figdir = figdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(figdir, exist_ok=True)

    angle_nb = angle_air_to_nb(angle_deg, nb)

    R_shapes = {name: [] for name in SHAPE_NAMES}
    T_shapes = {name: [] for name in SHAPE_NAMES}
    A_shapes = {name: [] for name in SHAPE_NAMES}
    thicknesses = []

    for thick in thickness_list:
        k_val = 40.0 / thick   # build_profiles: thickness = 40 / k_steep
        xx, e_re, shapes, _ = build_profiles(k_val, nb)
        thicknesses.append(thick)
        for name in SHAPE_NAMES:
            nc, dc = tmm_h.discretize_profile(xx, e_re + 1j * shapes[name], delta=delta)
            n_t = [nb] + list(nc) + [1.0]
            d_t = [np.inf] + list(dc) + [np.inf]
            T, R, A = tmm_h.TRA(n_t, d_t, lamb=wavelength, angle=angle_nb, pol=pol)
            R_shapes[name].append(R)
            T_shapes[name].append(T)
            A_shapes[name].append(A)
        print(f"  thick={thick:.3g} µm (k={k_val:.3g}): done")

    thicknesses = np.array(thicknesses)

    # Bare interface reference
    R_bare = R_bare_interface(nb, angle_deg, pol)

    base_fname = f'explore_thickness_single_{pol}_lam{wavelength}um_{int(angle_deg)}deg_delta{delta}'

    # Plot R
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.axhline(R_bare, color='gray', lw=1.5, ls=':', label='Bare interface', zorder=1)
    for name, col, mk in zip(SHAPE_NAMES, SHAPE_COLORS, SHAPE_MARKERS):
        ax.plot(thicknesses, R_shapes[name], f'{mk}-', color=col, lw=2, ms=6, label=name)
    ax.set_xlabel(r'Coating thickness ($\mu$m)')
    ax.set_ylabel(r'$R$')
    ax.set_title(f'Thickness sweep — {pol}-pol, λ={wavelength} µm, θ={angle_deg}°, δ={delta}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    fname_R = base_fname + '_R.png'
    plt.savefig(os.path.join(figdir, fname_R), dpi=150)
    print(f"  Saved {fname_R}")
    if show:
        plt.show()
    else:
        plt.close()

    # Plot 1 - A (absorption deficit — how far from perfect absorption)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    for name, col, mk in zip(SHAPE_NAMES, SHAPE_COLORS, SHAPE_MARKERS):
        deficit = 1.0 - np.array(A_shapes[name])
        ax.plot(thicknesses, deficit, f'{mk}-', color=col, lw=2, ms=6, label=name)
    ax.set_xlabel(r'Coating thickness ($\mu$m)')
    ax.set_ylabel(r'$1 - A$')
    ax.set_title(f'Absorption deficit — {pol}-pol, λ={wavelength} µm, θ={angle_deg}°, δ={delta}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    fname_A = base_fname + '_A.png'
    plt.savefig(os.path.join(figdir, fname_A), dpi=150)
    print(f"  Saved {fname_A}")
    if show:
        plt.show()
    else:
        plt.close()

    # Plot T
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    for name, col, mk in zip(SHAPE_NAMES, SHAPE_COLORS, SHAPE_MARKERS):
        ax.plot(thicknesses, T_shapes[name], f'{mk}-', color=col, lw=2, ms=6, label=name)
    ax.set_xlabel(r'Coating thickness ($\mu$m)')
    ax.set_ylabel(r'$T$')
    ax.set_title(f'Transmittance — {pol}-pol, λ={wavelength} µm, θ={angle_deg}°, δ={delta}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    fname_T = base_fname + '_T.png'
    plt.savefig(os.path.join(figdir, fname_T), dpi=150)
    print(f"  Saved {fname_T}")
    if show:
        plt.show()
    else:
        plt.close()

    return thicknesses, R_shapes, T_shapes, A_shapes, R_bare


def sweep_full_grid(thickness_list=None, lam_list=None, angle_list_deg=None,
                    nb=NB, delta=DELTA, figdir=None, show=False):
    """Full (λ, θ) grid sweep for both polarizations and all 5 loss shapes.

    For each combination of (thickness, λ, θ, pol, shape), computes R.
    Produces one figure per (pol, λ, θ) — 150 figures total with defaults.
    Raw data saved as .npy alongside the figures.

    Parameters
    ----------
    thickness_list : array — coating thicknesses in µm (default: 0.1–100, 10/decade)
    lam_list       : array — wavelengths in µm (default: 0.5–10 µm grid)
    angle_list_deg : array — angles in air in degrees (default: [0,30,45,60,80])
    """
    import time

    if thickness_list is None:
        thickness_list = THICKNESS_GRID
    if lam_list is None:
        lam_list = LAM_GRID
    if angle_list_deg is None:
        angle_list_deg = ANGLES_GRID

    figdir = figdir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'figures', 'April10')
    os.makedirs(figdir, exist_ok=True)

    n_thick  = len(thickness_list)
    n_lam    = len(lam_list)
    n_ang    = len(angle_list_deg)
    n_shapes = len(SHAPE_NAMES)
    pols     = ['s', 'p']

    # Precompute angles in n_b once (Snell's law, independent of thickness/shape/λ)
    angles_nb = {ang: angle_air_to_nb(ang, nb) for ang in angle_list_deg}

    # Bare interface references (precomputed)
    R_bare = {
        pol: {ang: R_bare_interface(nb, ang, pol)
              for ang in angle_list_deg}
        for pol in pols
    }

    # R_all[pol_idx, shape_idx, thick_idx, lam_idx, ang_idx]
    R_all = np.full((len(pols), n_shapes, n_thick, n_lam, n_ang), np.nan)

    k_values = 40.0 / np.array(thickness_list)

    print(f"\n=== Full grid sweep ===")
    print(f"  {n_thick} thicknesses × {n_lam} wavelengths × {n_ang} angles × "
          f"2 pols × {n_shapes} shapes = "
          f"{n_thick * n_lam * n_ang * 2 * n_shapes} TMM calls")
    print(f"  Figures → {figdir}\n")

    t_start = time.time()

    for ti, (k_val, thick) in enumerate(zip(k_values, thickness_list)):
        xx, e_re, shapes, _ = build_profiles(k_val, nb)

        for si, name in enumerate(SHAPE_NAMES):
            nc, dc = tmm_h.discretize_profile(xx, e_re + 1j * shapes[name], delta=delta)
            # Build full n/d lists once per (thickness, shape)
            n_t = [nb] + list(nc) + [1.0]
            d_t = [np.inf] + list(dc) + [np.inf]

            for pi, pol in enumerate(pols):
                for li, lam in enumerate(lam_list):
                    for ai, ang in enumerate(angle_list_deg):
                        T, R, A = tmm_h.TRA(n_t, d_t, lamb=lam,
                                            angle=angles_nb[ang], pol=pol)
                        R_all[pi, si, ti, li, ai] = R

        # Progress + ETA
        elapsed = time.time() - t_start
        frac    = (ti + 1) / n_thick
        eta_s   = elapsed / frac * (1 - frac)
        eta_str = f"{int(eta_s//60)}m{int(eta_s%60):02d}s" if ti > 0 else "—"
        print(f"  [{ti+1:2d}/{n_thick}] thick={thick:.3f} µm  (k={k_val:.2f})  "
              f"elapsed={elapsed:.1f}s  ETA={eta_str}", flush=True)

    # Save raw data
    data_path = os.path.join(figdir, 'sweep_grid_data.npy')
    np.save(data_path, R_all)
    meta = {
        'thickness_list': thickness_list,
        'lam_list':       lam_list,
        'angle_list_deg': angle_list_deg,
        'pols':           pols,
        'shape_names':    SHAPE_NAMES,
        'nb':             nb,
        'delta':          delta,
    }
    np.save(os.path.join(figdir, 'sweep_grid_meta.npy'), meta)
    print(f"\n  Data saved → {data_path}")

    # -------------------------------------------------------------------------
    # Figures: one per (pol, λ, θ) — all 5 shapes on the same axes
    # -------------------------------------------------------------------------
    print(f"\n  Saving {len(pols) * n_lam * n_ang} figures …")
    for pi, pol in enumerate(pols):
        for li, lam in enumerate(lam_list):
            for ai, ang in enumerate(angle_list_deg):
                fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
                ax.axhline(R_bare[pol][ang], color='gray', lw=1.5, ls=':',
                           label='Bare interface', zorder=1)
                for si, (name, col, mk) in enumerate(
                        zip(SHAPE_NAMES, SHAPE_COLORS, SHAPE_MARKERS)):
                    ax.plot(thickness_list, R_all[pi, si, :, li, ai],
                            f'{mk}-', color=col, lw=2, ms=6, label=name)
                ax.set_xlabel(r'Coating thickness ($\mu$m)')
                ax.set_ylabel(r'$R$')
                ax.set_title(f'{pol}-pol,  λ = {lam:.2f} µm,  θ = {ang:.0f}°')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.legend(fontsize=10, loc='best')
                plt.tight_layout()
                fname = f'grid_{pol}_lam{lam:05.2f}um_{int(ang):02d}deg.png'
                plt.savefig(os.path.join(figdir, fname), dpi=150)
                plt.close()

    total_s = time.time() - t_start
    print(f"\n  Done. Total time: {int(total_s//60)}m{int(total_s%60):02d}s")
    return R_all, thickness_list, lam_list, angle_list_deg


# ============================================================================
# CLI
# ============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Thickness sweep explorer for different loss shapes')
    parser.add_argument('--mode', choices=['wl_avg', 'ang_avg', 'single', 'all', 'grid'],
                        default='all', help='Which sweep to run (grid = full λ×θ sweep)')
    parser.add_argument('--angle', type=float, default=80, help='Angle (degrees)')
    parser.add_argument('--pol', type=str, default='s', help='Polarization (s or p)')
    parser.add_argument('--wavelength', type=float, default=3.0, help='Wavelength (µm)')
    parser.add_argument('--lam-min', type=float, default=2.0, help='Min wavelength (µm)')
    parser.add_argument('--lam-max', type=float, default=5.0, help='Max wavelength (µm)')
    parser.add_argument('--figdir', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    if args.mode in ('wl_avg', 'all'):
        print("\n=== Wavelength-averaged sweep ===")
        sweep_wavelength_averaged(args.angle, args.pol,
                                   lam_min=args.lam_min, lam_max=args.lam_max,
                                   figdir=args.figdir)

    if args.mode in ('ang_avg', 'all'):
        print("\n=== Angle-averaged sweep ===")
        sweep_angle_averaged(args.wavelength, args.pol,
                              figdir=args.figdir)

    if args.mode in ('single', 'all'):
        print("\n=== Single-point sweep ===")
        sweep_single(args.wavelength, args.angle, args.pol,
                      figdir=args.figdir)

    if args.mode == 'grid':
        sweep_full_grid(figdir=args.figdir)
