"""
Colorplots for thick sKK and GRIN coatings: 50 µm and 100 µm.

Produces two 2×2 figures (rows = s/p-pol, cols = 50/100 µm):
  - fig_colorplot_ratio_GRIN_over_sKK_thick.png  (GRIN/sKK ratio)
  - fig_colorplot_ratio_bulk_over_GRIN_thick.png  (bulk/GRIN ratio)

Saved to theory/figures/April12026/.
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

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 10, 'figure.dpi': 200, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'axes.linewidth': 1.2,
    'lines.linewidth': 2.0, 'mathtext.fontset': 'cm',
})

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'April12026')
os.makedirs(FIGDIR, exist_ok=True)


# ---- Sapphire data ----
def load_sapphire_data():
    ri_path = os.path.join(_PROJECT_ROOT, 'RI', 'lam_um_T_K_Al2O3_no_ko_ne_ke.dat')
    data = np.genfromtxt(ri_path)
    kdata = data[50:351, 3]
    ndata = data[50:351, 2]
    lamdata = data[50:351, 0]
    print(f"Loaded sapphire: λ = {lamdata[0]:.2f}–{lamdata[-1]:.2f} µm, {len(lamdata)} pts")
    return lamdata, ndata, kdata


# ---- TMM helpers (same as consolidated.py) ----
def Rback_2D(n_coating, d_coating, ndata, kdata, lamdata, angle_list_deg, pol):
    deg = np.pi / 180
    Rb = np.zeros((len(angle_list_deg), len(lamdata)))
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
    return Rb


def Rback_bulk_2D(ndata, kdata, lamdata, angle_list_deg, pol):
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
    pos = data_2d[data_2d > 0]
    if len(pos):
        gm = np.exp(np.mean(np.log(pos)))
        ax.text(0.97, 0.03, f'\u27e8\u00b7\u27e9={gm:.1e}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    nb = 1.7
    delta = 0.01
    color_angle_list = np.arange(0, 90, 1)
    color_pols = ['s', 'p']

    # k=0.8 → 50 µm,  k=0.4 → 100 µm
    k_vals = [0.8, 0.4]

    lamdata, ndata, kdata = load_sapphire_data()

    # ---- Bulk (coating-independent) ----
    Rb_bulk_2D = {}
    for pol_c in color_pols:
        print(f"  Computing bulk 2D ({pol_c}-pol)...")
        Rb_bulk_2D[pol_c] = Rback_bulk_2D(ndata, kdata, lamdata, color_angle_list, pol_c)

    # ---- sKK and GRIN for each thickness ----
    Rb_skk_2D = {pol_c: [] for pol_c in color_pols}
    Rb_grin_2D = {pol_c: [] for pol_c in color_pols}
    thicknesses = []

    for k_c in k_vals:
        dx_c = 1 / (100 * k_c)
        xmin_c = -20 / k_c
        xmax_c = -xmin_c
        nx_c = 1 + int(np.floor((xmax_c - xmin_c) / dx_c))
        xx_c = np.linspace(xmin_c, xmax_c, nx_c)
        e_re_c = tmm_h.logistic(xx_c, k_c, nb)
        e_im_c = tmm_h.ht_derivative(xx_c, e_re_c)
        thickness_c = xmax_c - xmin_c
        thicknesses.append(thickness_c)

        nc_skk_c, dc_skk_c = tmm_h.discretize_profile(xx_c, e_re_c + 1j * e_im_c, delta=delta)
        nc_grin_c, dc_grin_c = tmm_h.discretize_profile(xx_c, e_re_c + 0j, delta=delta)

        for pol_c in color_pols:
            print(f"  k={k_c} ({thickness_c:.0f} µm), {pol_c}-pol: sKK 2D...")
            Rb_s = Rback_2D(nc_skk_c, dc_skk_c, ndata, kdata, lamdata, color_angle_list, pol_c)
            Rb_skk_2D[pol_c].append(Rb_s)
            print(f"  k={k_c} ({thickness_c:.0f} µm), {pol_c}-pol: GRIN 2D...")
            Rb_g = Rback_2D(nc_grin_c, dc_grin_c, ndata, kdata, lamdata, color_angle_list, pol_c)
            Rb_grin_2D[pol_c].append(Rb_g)

    # ---- Figure 1: GRIN/sKK ratio (2×2) ----
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))
    for col, (k_c, thick_c) in enumerate(zip(k_vals, thicknesses)):
        for row, pol_c in enumerate(color_pols):
            ax = axes1[row, col]
            ratio = Rb_grin_2D[pol_c][col] / np.clip(Rb_skk_2D[pol_c][col], 1e-10, None)
            im = ax.pcolormesh(color_angle_list, lamdata, ratio.T,
                               norm=matplotlib.colors.LogNorm(vmin=1, vmax=1e4),
                               cmap='viridis', shading='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xlabel('AoI (degrees)')
            ax.set_ylabel(r'Wavelength ($\mu$m)')
            ax.set_title(f'R_GRIN/R_sKK — {thick_c:.0f} µm, {pol_c}-pol')
            _annotate_geomean(ax, ratio)
    plt.tight_layout()
    out1 = os.path.join(FIGDIR, 'fig_colorplot_ratio_GRIN_over_sKK_thick.png')
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"  Saved {out1}")

    # ---- Figure 2: bulk/GRIN ratio (2×2) ----
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    for col, (k_c, thick_c) in enumerate(zip(k_vals, thicknesses)):
        for row, pol_c in enumerate(color_pols):
            ax = axes2[row, col]
            ratio = Rb_bulk_2D[pol_c] / np.clip(Rb_grin_2D[pol_c][col], 1e-10, None)
            im = ax.pcolormesh(color_angle_list, lamdata, ratio.T,
                               norm=matplotlib.colors.LogNorm(vmin=1, vmax=1e4),
                               cmap='viridis', shading='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xlabel('AoI (degrees)')
            ax.set_ylabel(r'Wavelength ($\mu$m)')
            ax.set_title(f'R_bulk/R_GRIN — {thick_c:.0f} µm, {pol_c}-pol')
            _annotate_geomean(ax, ratio)
    plt.tight_layout()
    out2 = os.path.join(FIGDIR, 'fig_colorplot_ratio_bulk_over_GRIN_thick.png')
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"  Saved {out2}")

    print("\nDone.")
