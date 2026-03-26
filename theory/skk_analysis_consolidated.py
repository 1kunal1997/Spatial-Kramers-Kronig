"""
Spatial KK Analysis — Consolidated Script
==========================================
All calculations and figure generation for the sKK AR coating paper.
Generates all 10 figures used in the presentation slides.

Figures:
  1. fig_new1_lorentz_vs_grin.png  — Lorentzian HT + GRIN periodic continuation (2-panel)
  2. fig0a_ht_problem.png          — Asymmetric endpoint problem (single curve: direct FFT)
  3. fig_new3_derivative_result.png — Derivative-then-integrate result (no textbox in a, lossy air in b)
  4. fig3_reflection_wavelength.png — Backside reflection: sKK vs GRIN vs Bulk
  5. fig_fom_intro_single.png      — Spectral FoM introduction (single panel)
  6. fig4_alpha_tradeoff.png       — R-A tradeoff with spectral FoM on twin axis
  7. fig5_sigma_gating.png         — Sigma gating: profiles + reflection (no bulk curve)
  8. fig_new5_fom_explanation.png   — FoM comparison: full sKK vs gated (both panels)
  9. fig6_angle_resolved.png       — Angle-resolved backside reflection
 10. fig9_thickness_single.png     — Thickness design space (single panel)

Requirements:
  - numpy, scipy, matplotlib, tmm
  - Sapphire data: lam_um_T_K_Al2O3_no_ko_ne_ke.dat

NOTE: This script intentionally re-implements core functions from tmm_helper.py
to remain self-contained for paper figure reproduction. If you change the physics
in tmm_helper.py, sync the corresponding functions here if needed.
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.signal.windows import tukey
from scipy.integrate import cumulative_trapezoid
import tmm

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

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ============================================================================
# Sapphire optical constants (realistic mid-IR)
# ============================================================================
def load_sapphire_data():
    """Load sapphire ordinary-ray optical constants (2-5 μm)."""
    ri_path = os.path.join(_PROJECT_ROOT, 'RI', 'lam_um_T_K_Al2O3_no_ko_ne_ke.dat')
    for ri_path in [ri_path]:
        if os.path.exists(ri_path):
            data = np.genfromtxt(ri_path)
            # columns: lam(μm), T(K), n_o, k_o, n_e, k_e
            kdata = data[50:351, 3]
            ndata = data[50:351, 2]
            lamdata = data[50:351, 0]
            print(f"Loaded sapphire data: λ = {lamdata[0]:.2f}–{lamdata[-1]:.2f} μm, "
                  f"{len(lamdata)} pts, n = {ndata.min():.4f}–{ndata.max():.4f}")
            return lamdata, ndata, kdata
    raise FileNotFoundError("Sapphire data file not found")

# ============================================================================
# Core physics functions
# ============================================================================
def logistic_eps(x, k, nb, sx=1):
    """Logistic GRIN dielectric profile: ε'(x) = (nb²-1)/(1+exp(sk·x)) + 1"""
    return (nb**2 - 1) / (1 + np.exp(sx * k * x)) + 1

def eps_lorentz(x, A, x0, nb):
    """Lorentzian dielectric profile: ε(x) = nb² - A·x₀/(x + i·x₀)"""
    return nb**2 - A * x0 / (x + 1j * x0)

def ht_derivative(xx, e_re, pad_factor=8):
    """Derivative-then-integrate HT method (main contribution of the paper).

    Key idea: dε'/dx → 0 at both ends, so standard FFT-based HT works
    on the derivative. Integrate back to recover ε''.
    """
    N = len(e_re)
    u = np.gradient(e_re, xx)
    pad = pad_factor * N
    u_pad = np.pad(u, (pad, pad), mode='constant', constant_values=0)
    v_pad = np.imag(hilbert(u_pad))
    v = v_pad[pad:pad+N]
    e_im = cumulative_trapezoid(v, xx, initial=0)
    e_im -= np.linspace(e_im[0], e_im[-1], N)
    return e_im

def smooth_gate(eps_re, eps0, sigma):
    """Smooth tanh gate: g(ε') = 0.5*(1 + tanh((ε' - ε₀)/σ))"""
    return 0.5 * (1.0 + np.tanh((eps_re - eps0) / sigma))

def discretize_profile(xx, ee, delta=0.05):
    """Discretize continuous ε(x) into TMM layers."""
    e_scale = np.max(np.abs(ee - ee[0]))
    if e_scale < 1e-12: e_scale = 1.0
    x_scale = xx[-1] - xx[0]
    xq, eq = [xx[0]], [ee[0]]
    for k in range(1, len(xx)):
        dx = (xx[k] - xq[-1]) / x_scale
        de = abs(ee[k] - eq[-1]) / e_scale
        ds = np.sqrt(dx**2 + de**2)
        if ds > delta:
            xq.append(xx[k]); eq.append(ee[k])
    xq.append(xx[-1]); eq.append(ee[-1])
    xq, eq = np.array(xq), np.array(eq)
    d_list = np.diff(xq).tolist()
    e_list = (eq[:-1] + eq[1:]) / 2
    n_list = np.sqrt(e_list).tolist()
    return n_list, d_list

def hilbert_fom(x, u, v, pad_factor=8):
    """Hilbert FoM: derivative-space correlation between ε'' and ideal HT-derived ε''."""
    ud = np.gradient(u, x)
    vd = np.gradient(v, x)
    pad = pad_factor * len(x)
    ud_pad = np.pad(ud, (pad, pad), mode='constant')
    vd_ht = np.imag(hilbert(ud_pad))[pad:-pad]
    num = 2 * np.trapezoid(vd * vd_ht, x)
    den = np.trapezoid(vd**2, x) + np.trapezoid(vd_ht**2, x)
    return 100 * max(0.0, num / den)

def spectral_fom(x, u, v, pad_factor=8):
    """Spectral one-sidedness FoM.

    Forms z(x) = dε'/dx + i·dε''/dx, computes Z(k) = FFT{z(x)},
    then measures: FoM = (E₊ - E₋)/(E₊ + E₋)
    where E₊ = ∫|Z(k)|² dk for k > 0, E₋ for k < 0.

    A truly analytic (sKK-consistent) profile has FoM → 1.
    """
    ud = np.gradient(u, x)
    vd = np.gradient(v, x)
    z = ud + 1j * vd
    pad = pad_factor * len(x)
    z = np.pad(z, (pad, pad), mode='constant')
    dx = x[1] - x[0]
    Z = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(z)))
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(z), d=dx))
    E_pos = np.sum(np.abs(Z[k > 0])**2)
    E_neg = np.sum(np.abs(Z[k < 0])**2)
    return 100 * max(0.0, (E_pos - E_neg) / (E_pos + E_neg))

# ============================================================================
# TMM helpers
# ============================================================================
def Rback_vs_wavelength(n_coating, d_coating, ndata, kdata, lamdata, angle_deg, pol):
    """Backside reflection and absorption vs wavelength (incoherent substrate)."""
    deg = np.pi/180; angle = angle_deg * deg
    n_t = list(n_coating); d_t = list(d_coating)
    c_t = ['c'] * len(n_coating)
    n_t.insert(0, complex(ndata[0], kdata[0]))
    d_t.insert(0, 5000); c_t.insert(0, 'i')
    n_t.insert(0, 1); n_t.append(1)
    d_t.insert(0, np.inf); d_t.append(np.inf)
    c_t.insert(0, 'i'); c_t.append('i')

    Rb = np.zeros(len(lamdata)); At = np.zeros(len(lamdata))
    for i, wl in enumerate(lamdata):
        n_t[1] = complex(ndata[i], kdata[i])
        th_f = tmm.snell(1, n_t[1], angle)
        Rf = tmm.interface_R(pol, 1, n_t[1], angle, th_f)
        res = tmm.inc_tmm(pol, n_t, d_t, c_t, angle, wl)
        Rb[i] = res['R'] - Rf
        At[i] = 1 - res['T'] - res['R']
    return Rb, At

def Rback_vs_angle(n_coating, d_coating, n_sub, angle_list_deg, lam, pol):
    """Backside reflection vs angle at a single wavelength."""
    deg = np.pi/180
    n_t = list(n_coating); d_t = list(d_coating)
    c_t = ['c'] * len(n_coating)
    n_t.insert(0, n_sub); d_t.insert(0, 5000); c_t.insert(0, 'i')
    n_t.insert(0, 1); n_t.append(1)
    d_t.insert(0, np.inf); d_t.append(np.inf)
    c_t.insert(0, 'i'); c_t.append('i')

    Rb = np.zeros(len(angle_list_deg)); At = np.zeros(len(angle_list_deg))
    for i, ang in enumerate(angle_list_deg):
        theta = ang * deg
        th_f = tmm.snell(1, n_sub, theta)
        Rf = tmm.interface_R(pol, 1, n_sub, theta, th_f)
        res = tmm.inc_tmm(pol, n_t, d_t, c_t, theta, lam)
        Rb[i] = res['R'] - Rf
        At[i] = 1 - res['T'] - res['R']
    return Rb, At

def Rback_bulk_wl(ndata, kdata, lamdata, angle_deg, pol):
    """Bulk sapphire backside reflection (no coating)."""
    deg = np.pi/180; angle = angle_deg * deg
    n_b = [1, complex(ndata[0], kdata[0]), 1]
    d_b = [np.inf, 5000, np.inf]; c_b = ['i','i','i']
    Rb = np.zeros(len(lamdata)); At = np.zeros(len(lamdata))
    for i, wl in enumerate(lamdata):
        n_b[1] = complex(ndata[i], kdata[i])
        th_f = tmm.snell(1, n_b[1], angle)
        Rf = tmm.interface_R(pol, 1, n_b[1], angle, th_f)
        res = tmm.inc_tmm(pol, n_b, d_b, c_b, angle, wl)
        Rb[i] = res['R'] - Rf; At[i] = 1 - res['T'] - res['R']
    return Rb, At

def Rback_bulk_angle(n_sub, angle_list_deg, lam, pol):
    """Bulk sapphire backside reflection vs angle."""
    deg = np.pi/180
    n_b = [1, n_sub, 1]; d_b = [np.inf, 5000, np.inf]; c_b = ['i','i','i']
    Rb = np.zeros(len(angle_list_deg))
    for i, ang in enumerate(angle_list_deg):
        theta = ang * deg
        th_f = tmm.snell(1, n_sub, theta)
        Rf = tmm.interface_R(pol, 1, n_sub, theta, th_f)
        res = tmm.inc_tmm(pol, n_b, d_b, c_b, theta, lam)
        Rb[i] = res['R'] - Rf
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


# ============================================================================
# MAIN — Generate all figures
# ============================================================================
if __name__ == '__main__':

    # ---- Setup ----
    k_steep = 100; nb = 1.7; delta = 0.01
    dx = 1/(100*k_steep); xmin = -20/k_steep; xmax = -xmin
    nx = 1 + int(np.floor((xmax - xmin) / dx))
    xx = np.linspace(xmin, xmax, nx)
    e_re = logistic_eps(xx, k_steep, nb)

    lamdata, ndata, kdata = load_sapphire_data()

    # ht_original() moved to archive/tmm_helper_backup_v2.py
    e_im_deriv = ht_derivative(xx, e_re)

    angle_test = 80; pol_test = 's'

    # Coatings
    ee_full = e_re + 1j * e_im_deriv
    nc_full, dc_full = discretize_profile(xx, ee_full, delta=delta)
    ee_grin = e_re + 0j
    nc_grin, dc_grin = discretize_profile(xx, ee_grin, delta=delta)

    # TMM baseline results
    Rb_full, A_full = Rback_vs_wavelength(nc_full, dc_full, ndata, kdata, lamdata, angle_test, pol_test)
    Rb_grin, A_grin = Rback_vs_wavelength(nc_grin, dc_grin, ndata, kdata, lamdata, angle_test, pol_test)
    Rb_bulk, A_bulk = Rback_bulk_wl(ndata, kdata, lamdata, angle_test, pol_test)

    nk_d = np.sqrt(e_re + 1j * e_im_deriv)
    mask_lossy = (np.real(nk_d) < 1.15) & (np.imag(nk_d) > 0.01)

    print("=== FoM Comparison ===")
    for name, eim in [("Derivative", e_im_deriv)]:
        hf = hilbert_fom(xx, e_re, eim)
        sf = spectral_fom(xx, e_re, eim)
        print(f"  {name}: HT_FoM={hf:.2f}%, Spectral_FoM={sf:.2f}%")

    Rb_avg_bulk = np.trapezoid(Rb_bulk, lamdata) / (lamdata[-1] - lamdata[0])
    Rb_avg_grin = np.trapezoid(Rb_grin, lamdata) / (lamdata[-1] - lamdata[0])
    Rb_avg_full = np.trapezoid(Rb_full, lamdata) / (lamdata[-1] - lamdata[0])
    A_avg_full = np.trapezoid(A_full, lamdata) / (lamdata[-1] - lamdata[0])
    print(f"\n=== Wavelength sweep ({pol_test}-pol, {angle_test}°) ===")
    print(f"  Bulk:     ⟨R_back⟩ = {Rb_avg_bulk:.5f}")
    print(f"  GRIN:     ⟨R_back⟩ = {Rb_avg_grin:.5f}")
    print(f"  sKK full: ⟨R_back⟩ = {Rb_avg_full:.5f}, ⟨A⟩ = {A_avg_full:.5f}")

    # ====================================================================
    # FIGURE 1 (Slide 1): Lorentzian HT + GRIN periodic continuation
    # 2-panel: (a) Lorentzian with x₀/A notation, (b) GRIN periodic continuation
    # ====================================================================
    A_lor = 0.5; x0_lor = 0.05
    dx_l = x0_lor / 50; xmax_l = x0_lor * 100
    nx_l = 1 + int(np.floor(2 * xmax_l / dx_l))
    xx_l = np.linspace(-xmax_l, xmax_l, nx_l)
    ee_l = eps_lorentz(xx_l, A_lor, x0_lor, nb)

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

    # (b) GRIN periodic continuation → jump discontinuity
    e_tiled = np.tile(e_re, 3)
    xx_tiled = np.concatenate([xx - (xmax-xmin), xx, xx + (xmax-xmin)])
    ax2.plot(xx_tiled, e_tiled, color=BLUE, lw=2)
    ax2.axvline(xmin, color='gray', lw=1, ls='--', alpha=0.5)
    ax2.axvline(xmax, color='gray', lw=1, ls='--', alpha=0.5)
    ax2.axvspan(xmin, xmax, alpha=0.06, color='blue')
    ax2.text((xmin+xmax)/2, 2.4, 'One period', fontsize=10, ha='center', color=BLUE,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor=BLUE, boxstyle='round'))
    ax2.annotate('Jump\ndiscontinuity', xy=(xmax, 1.5),
                 xytext=(xmax + 0.12, 2.0), fontsize=10, ha='center',
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
                 color=RED, bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round'))
    ax2.set_xlabel(r'$x$ ($\mu$m)'); ax2.set_ylabel(r"$\epsilon'(x)$")
    ax2.set_title(r"(b) GRIN: FFT periodic continuation $\to$ jump discontinuity")
    ax2.set_xlim(xmin - 0.25, xmax + 0.35)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig_new1_lorentz_vs_grin.png')
    plt.close()
    print("Saved fig 1: Lorentzian vs GRIN")

    # ====================================================================
    # FIGURE 2 (Slide 2): Asymmetric endpoint problem — single curve
    # ====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) GRIN profile with endpoint labels
    ax = axes[0]
    ax.plot(xx, e_re, color=BLUE, lw=2.5)
    ax.axhline(nb**2, color='gray', lw=1, ls=':', alpha=0.5)
    ax.axhline(1.0, color='gray', lw=1, ls=':', alpha=0.5)
    ax.annotate(r'$\epsilon_b = n_b^2 = %.2f$' % nb**2, xy=(xmin, nb**2),
                xytext=(xmin + 0.02, nb**2 + 0.08), fontsize=11, color='gray')
    ax.annotate(r'$\epsilon_{\rm air} = 1$', xy=(xmax, 1.0),
                xytext=(xmax - 0.08, 1.15), fontsize=11, color='gray')
    ax.annotate('', xy=(0.15, 1.0), xytext=(0.15, nb**2),
                arrowprops=dict(arrowstyle='<->', color=RED, lw=2))
    ax.text(0.165, (1 + nb**2)/2, r'$\Delta\epsilon = %.2f$' % (nb**2 - 1),
            fontsize=12, color=RED, va='center')
    ax.set_xlabel(r'$x$ ($\mu$m)'); ax.set_ylabel(r"$\epsilon'(x)$")
    ax.set_title(r"(a) GRIN profile: different endpoints ($\epsilon_b \neq \epsilon_{\rm air}$)")

    # (b) Direct FFT HT — single orange curve only
    ax = axes[1]
    z_naive = hilbert(e_re)
    e_im_naive = np.imag(z_naive)
    ax.plot(xx, e_im_naive, color=ORANGE, lw=2, label='No padding (direct FFT)')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.annotate('Edge artifacts from\nendpoint mismatch',
                xy=(xmin + 0.02, e_im_naive[10]), xytext=(0.0, max(e_im_naive)*0.8),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round'))
    ax.set_xlabel(r'$x$ ($\mu$m)'); ax.set_ylabel(r"$\epsilon''(x)$")
    ax.set_title(r"(b) Naive Hilbert transform — endpoint artifacts")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig0a_ht_problem.png')
    plt.close()
    print("Saved fig 2: Asymmetric endpoint problem")

    # ====================================================================
    # FIGURE 3 (Slide 3): Derivative-then-integrate result
    # No textbox in (a), lossy air textbox shifted up in (b)
    # ====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Key insight: derivatives → 0 at both ends
    ax = axes[0]; ax_twin = ax.twinx()
    u_deriv = np.gradient(e_re, xx)
    pad_n = 8 * len(xx)
    v_deriv_ht = np.imag(hilbert(np.pad(u_deriv, (pad_n, pad_n),
                                         mode='constant')))[pad_n:pad_n+len(xx)]
    ax.plot(xx, e_re, color=BLUE, lw=2.5, label=r"$\epsilon'(x)$")
    ax.plot(xx, u_deriv, color=ORANGE, lw=2, label=r"$d\epsilon'/dx$")
    ax_twin.plot(xx, v_deriv_ht, color=RED, lw=2, label=r"$\mathcal{H}[d\epsilon'/dx]$")
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r"$\epsilon'$, $d\epsilon'/dx$", color=BLUE)
    ax_twin.set_ylabel(r"$\mathcal{H}[d\epsilon'/dx]$", color=RED)
    ax.set_title(r"(a) Key insight: $d\epsilon'/dx \to 0$ at both ends")
    ax.tick_params(axis='y', labelcolor=BLUE)
    ax_twin.tick_params(axis='y', labelcolor=RED)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=9)

    # (b) Result: ε' and ε'' with lossy air shaded
    ax = axes[1]; ax2 = ax.twinx()
    ax.plot(xx, e_re, color=BLUE, lw=2.5)
    ax2.plot(xx, e_im_deriv, color=RED, lw=2.5)
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r"$\epsilon'$", color=BLUE)
    ax2.set_ylabel(r"$\epsilon''$", color=RED)
    ax.set_title(r"(b) Result: $\epsilon(x) = \epsilon' + i\epsilon''$ (derivative method)")
    ax.tick_params(axis='y', labelcolor=BLUE)
    ax2.tick_params(axis='y', labelcolor=RED)

    # Shade lossy air and place label high enough to avoid red curve
    lossy_mask_b = (e_re < 1.1) & (e_im_deriv > 0.01)
    if np.any(lossy_mask_b):
        idx_start = np.where(lossy_mask_b)[0][0]
        idx_end = np.where(lossy_mask_b)[0][-1]
        ax.axvspan(xx[idx_start], xx[idx_end], alpha=0.15, color='orange', zorder=0)
        y_mid = 0.75 * ax.get_ylim()[1] + 0.25 * ax.get_ylim()[0]
        x_mid = 0.5 * (xx[idx_start] + xx[idx_end])
        ax.text(x_mid, y_mid, '"Lossy air"\n' + r'$n \approx 1, k > 0$',
                fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='wheat', alpha=0.8, boxstyle='round'))

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig_new3_derivative_result.png')
    plt.close()
    print("Saved fig 3: Derivative result")

    # ====================================================================
    # FIGURE 4 (Slide 4): Backside reflection comparison
    # Legend: Bulk sapphire / GRIN coating / sKK coating
    # ====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(lamdata, Rb_bulk, color='gray', lw=2, label='Bulk sapphire')
    ax.plot(lamdata, Rb_grin, '--', color=BLUE, lw=2, label='GRIN coating')
    ax.plot(lamdata, Rb_full, color=GREEN, lw=2.5, label='sKK coating')
    ax.set_xlabel(r'Wavelength ($\mu$m)'); ax.set_ylabel(r'$R_{\rm back}$')
    ax.set_title(f'(a) Backside reflection ({pol_test}-pol, {angle_test}°)')
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.plot(lamdata, A_bulk, color='gray', lw=2, label='Bulk sapphire')
    ax.plot(lamdata, A_grin, '--', color=BLUE, lw=2, label='GRIN coating')
    ax.plot(lamdata, A_full, color=RED, lw=2.5, label='sKK coating')
    ax.set_xlabel(r'Wavelength ($\mu$m)'); ax.set_ylabel('Absorbance')
    ax.set_title(f'(b) Absorption ({pol_test}-pol, {angle_test}°)')
    ax.legend(fontsize=10, loc='center left')

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig3_reflection_wavelength.png')
    plt.close()
    print("Saved fig 4: Backside reflection")

    # ====================================================================
    # FIGURE 5 (Slide 5): Spectral FoM introduction — single panel
    # y-axis: |Z(k)|² (arb. units)
    # ====================================================================
    # Use denser grid for smoother spectral plot
    xx_dense = np.linspace(-0.2, 0.2, 4001)
    e_re_dense = (nb**2 - 1) / (1 + np.exp(k_steep * xx_dense)) + 1
    e_im_dense = ht_derivative(xx_dense, e_re_dense)

    ud_d = np.gradient(e_re_dense, xx_dense)
    vd_d = np.gradient(e_im_dense, xx_dense)
    z_d = ud_d + 1j * vd_d
    pad_d = 8 * len(xx_dense)
    z_d_pad = np.pad(z_d, (pad_d, pad_d), mode='constant')
    dx_d = xx_dense[1] - xx_dense[0]
    Z_d = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(z_d_pad)))
    k_freq = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(z_d_pad), d=dx_d))
    power_d = np.abs(Z_d)**2
    power_d /= power_d.max()

    E_pos = np.sum(np.abs(Z_d[k_freq > 0])**2)
    E_neg = np.sum(np.abs(Z_d[k_freq <= 0])**2)
    fom_full = (E_pos - E_neg) / (E_pos + E_neg)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))
    ax.fill_between(k_freq[k_freq >= 0], power_d[k_freq >= 0], alpha=0.35, color='#2ca02c',
                    label=r'Positive $k$')
    ax.fill_between(k_freq[k_freq <= 0], power_d[k_freq <= 0], alpha=0.35, color='#d62728',
                    label=r'Negative $k$ (forbidden)')
    ax.plot(k_freq, power_d, 'k-', linewidth=0.5, alpha=0.5)
    ax.set_yscale('log'); ax.set_ylim(1e-10, 2); ax.set_xlim(-400, 400)
    ax.set_xlabel(r'Spatial frequency $k$ ($\mu$m$^{-1}$)')
    ax.set_ylabel(r'$|Z(k)|^2$ (arb. units)')
    ax.set_title(f'Full sKK profile — Spectral FoM = {fom_full*100:.1f}%')
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig_fom_intro_single.png')
    plt.close()
    print("Saved fig 5: FoM intro (single panel)")

    # ====================================================================
    # FIGURE 6 (Slide 6): R-A tradeoff with spectral FoM on twin axis
    # ====================================================================
    alpha_list = np.linspace(0, 1, 40)
    R_vs_a = np.zeros(len(alpha_list))
    A_vs_a = np.zeros(len(alpha_list))
    FoM_vs_a = np.zeros(len(alpha_list))

    for j, alpha in enumerate(alpha_list):
        e_im_a = alpha * e_im_deriv
        ee_a = e_re + 1j * e_im_a
        nc_a, dc_a = discretize_profile(xx, ee_a, delta=delta)
        Rb_a, At_a = Rback_vs_wavelength(nc_a, dc_a, ndata, kdata, lamdata, angle_test, pol_test)
        R_vs_a[j] = np.trapezoid(Rb_a, lamdata) / (lamdata[-1] - lamdata[0])
        A_vs_a[j] = np.trapezoid(At_a, lamdata) / (lamdata[-1] - lamdata[0])
        FoM_vs_a[j] = spectral_fom(xx, e_re, e_im_a) / 100.0
        if j % 10 == 0:
            print(f"  α={alpha:.2f}: ⟨R⟩={R_vs_a[j]:.4f}, ⟨A⟩={A_vs_a[j]:.4f}, FoM={FoM_vs_a[j]:.3f}")

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
    plt.savefig(f'{FIGDIR}/fig4_alpha_tradeoff.png')
    plt.close()
    print("Saved fig 6: Alpha tradeoff")

    # ====================================================================
    # FIGURE 7 (Slide 7): Sigma gating — no bulk curve, tight layout
    # ====================================================================
    sigma_list = [None, 0.1, 0.01]
    n0_gate = 1.3

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    fig.subplots_adjust(hspace=0.35, wspace=0.45)

    for idx, sigma in enumerate(sigma_list):
        e_im_base = e_im_deriv.copy()
        if sigma is not None:
            gate = smooth_gate(e_re, n0_gate**2, sigma)
            e_im_gated = e_im_base * gate
            label = f'$\\sigma$ = {sigma}'
        else:
            e_im_gated = e_im_base
            label = r'No gating ($\alpha=1$)'

        # Top row: dielectric function
        ax = axes[0, idx]; ax2t = ax.twinx()
        ax.plot(xx, e_re, color=BLUE, lw=2.5)
        ax2t.plot(xx, e_im_gated, color=RED, lw=2.5)
        ax.set_xlabel(r'$x$ ($\mu$m)'); ax.set_ylabel(r"$\epsilon'$", color=BLUE)
        ax2t.set_ylabel(r"$\epsilon''$", color=RED)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelcolor=BLUE)
        ax2t.tick_params(axis='y', labelcolor=RED)
        sf = spectral_fom(xx, e_re, e_im_gated)
        ax.text(0.97, 0.97, f'Spectral FoM:\n{sf:.1f}%',
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(facecolor='wheat', alpha=0.7, boxstyle='round'))

        # Bottom row: reflection (no bulk curve)
        ax = axes[1, idx]
        ee_g = e_re + 1j * e_im_gated
        nc_g, dc_g = discretize_profile(xx, ee_g, delta=delta)
        Rb_g, At_g = Rback_vs_wavelength(nc_g, dc_g, ndata, kdata, lamdata, angle_test, pol_test)
        ax.plot(lamdata, Rb_g, color=GREEN, lw=2.5, label=r'$R_{\rm back}$')
        ax.plot(lamdata, At_g, color=RED, lw=2.5, label='$A$')
        ax.set_xlabel(r'Wavelength ($\mu$m)'); ax.set_ylabel('Fraction of Power')
        R_avg = np.trapezoid(Rb_g, lamdata)/(lamdata[-1]-lamdata[0])
        A_avg = np.trapezoid(At_g, lamdata)/(lamdata[-1]-lamdata[0])
        ax.set_title(f'$\\langle R \\rangle$={R_avg:.4f}, $\\langle A \\rangle$={A_avg:.4f}')
        ax.legend(fontsize=9)

    plt.savefig(f'{FIGDIR}/fig5_sigma_gating.png')
    plt.close()
    print("Saved fig 7: Sigma gating")

    # ====================================================================
    # FIGURE 8 (Slide 8): FoM comparison — full sKK vs gated (both panels)
    # y-axis: |Z(k)|² (arb. units)
    # ====================================================================
    gate_01 = smooth_gate(e_re_dense, 1.3**2, 0.1)
    e_im_gated_dense = e_im_dense * gate_01

    ud_g = np.gradient(e_re_dense, xx_dense)
    vd_g = np.gradient(e_im_gated_dense, xx_dense)
    z_g = ud_g + 1j * vd_g
    z_g_pad = np.pad(z_g, (pad_d, pad_d), mode='constant')
    Z_g = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(z_g_pad)))
    power_g = np.abs(Z_g)**2; power_g /= power_g.max()
    E_pos_g = np.sum(np.abs(Z_g[k_freq > 0])**2)
    E_neg_g = np.sum(np.abs(Z_g[k_freq <= 0])**2)
    fom_gated = (E_pos_g - E_neg_g) / (E_pos_g + E_neg_g)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, pwr, fom_val, title_prefix in [
        (ax1, power_d, fom_full, '(a) Full sKK profile'),
        (ax2, power_g, fom_gated, r'(b) Gated ($\sigma=0.1$)')]:
        ax.fill_between(k_freq[k_freq >= 0], pwr[k_freq >= 0], alpha=0.35, color='#2ca02c',
                        label=r'Positive $k$')
        ax.fill_between(k_freq[k_freq <= 0], pwr[k_freq <= 0], alpha=0.35, color='#d62728',
                        label=r'Negative $k$')
        ax.plot(k_freq, pwr, 'k-', linewidth=0.5, alpha=0.5)
        ax.set_yscale('log'); ax.set_ylim(1e-10, 2); ax.set_xlim(-400, 400)
        ax.set_xlabel(r'Spatial frequency $k$ ($\mu$m$^{-1}$)')
        ax.set_ylabel(r'$|Z(k)|^2$ (arb. units)')
        ax.set_title(f'{title_prefix} — Spectral FoM = {fom_val*100:.1f}%')
        ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig_new5_fom_explanation.png')
    plt.close()
    print("Saved fig 8: FoM comparison")

    # ====================================================================
    # FIGURE 9 (Slide 9): Angle-resolved backside reflection
    # Legend: Bulk sapphire / GRIN coating / sKK coating
    # ====================================================================
    angle_list = np.arange(0, 90, 1)
    lam_test = 3.0
    idx_lam = np.argmin(np.abs(lamdata - lam_test))
    n_sub = complex(ndata[idx_lam], kdata[idx_lam])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for pidx, pol in enumerate(['s', 'p']):
        ax = axes[pidx]
        Rb_bulk_a = Rback_bulk_angle(n_sub, angle_list, lam_test, pol)
        Rb_grin_a, _ = Rback_vs_angle(nc_grin, dc_grin, n_sub, angle_list, lam_test, pol)
        Rb_skk_a, _ = Rback_vs_angle(nc_full, dc_full, n_sub, angle_list, lam_test, pol)
        ax.plot(angle_list, Rb_bulk_a, color='gray', lw=2, label='Bulk sapphire')
        ax.plot(angle_list, Rb_grin_a, '--', color=BLUE, lw=2, label='GRIN coating')
        ax.plot(angle_list, Rb_skk_a, color=GREEN, lw=2.5, label='sKK coating')
        ax.set_xlabel('Angle of Incidence (°)')
        ax.set_ylabel(r'$R_{\rm back}$')
        ax.set_title(f'{pol}-polarization, $\\lambda$ = {lam_test} $\\mu$m')
        ax.legend(fontsize=10); ax.set_xlim(0, 89)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig6_angle_resolved.png')
    plt.close()
    print("Saved fig 9: Angle-resolved")

    # ====================================================================
    # FIGURE 10 (Slide 10): Thickness design space — single panel
    # Legend: Bulk sapphire / GRIN coating / sKK coating / sKK gated (σ=0.1)
    # ====================================================================
    k_values = np.array([2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100])
    thicknesses = []
    R_grin_arr, R_skk_arr, R_gated_arr = [], [], []
    A_skk_arr = []

    for k_val in k_values:
        dx_k = 1/(100*k_val); xmin_k = -20/k_val; xmax_k = -xmin_k
        nx_k = 1+int(np.floor((xmax_k-xmin_k)/dx_k))
        xx_k = np.linspace(xmin_k, xmax_k, nx_k)
        e_re_k = logistic_eps(xx_k, k_val, nb)
        e_im_k = ht_derivative(xx_k, e_re_k)
        thickness = 2*xmax_k
        thicknesses.append(thickness)

        # GRIN
        nc_gk, dc_gk = discretize_profile(xx_k, e_re_k + 0j, delta=delta)
        Rb_gk, _ = Rback_vs_wavelength(nc_gk, dc_gk, ndata, kdata, lamdata, angle_test, pol_test)
        R_grin_arr.append(np.trapezoid(Rb_gk, lamdata) / (lamdata[-1] - lamdata[0]))

        # Full sKK
        nc_fk, dc_fk = discretize_profile(xx_k, e_re_k + 1j*e_im_k, delta=delta)
        Rb_fk, At_fk = Rback_vs_wavelength(nc_fk, dc_fk, ndata, kdata, lamdata, angle_test, pol_test)
        R_skk_arr.append(np.trapezoid(Rb_fk, lamdata) / (lamdata[-1] - lamdata[0]))
        A_skk_arr.append(np.trapezoid(At_fk, lamdata) / (lamdata[-1] - lamdata[0]))

        # Gated sKK (σ=0.1)
        gate_k = smooth_gate(e_re_k, 1.3**2, 0.1)
        nc_gatk, dc_gatk = discretize_profile(xx_k, e_re_k + 1j*e_im_k*gate_k, delta=delta)
        Rb_gatk, _ = Rback_vs_wavelength(nc_gatk, dc_gatk, ndata, kdata, lamdata, angle_test, pol_test)
        R_gated_arr.append(np.trapezoid(Rb_gatk, lamdata) / (lamdata[-1] - lamdata[0]))

        print(f"  k={k_val:3d} ({thickness:.1f} μm): R_GRIN={R_grin_arr[-1]:.4f}, "
              f"R_sKK={R_skk_arr[-1]:.4f}, R_gated={R_gated_arr[-1]:.4f}")

    thicknesses = np.array(thicknesses)
    R_grin_arr = np.array(R_grin_arr)
    R_skk_arr = np.array(R_skk_arr)
    R_gated_arr = np.array(R_gated_arr)
    A_skk_arr = np.array(A_skk_arr)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    ax.axhline(Rb_avg_bulk, color='gray', lw=1.5, ls=':', label='Bulk sapphire', zorder=1)
    ax.plot(thicknesses, R_grin_arr, 'o-', color=BLUE, lw=2.5, ms=7, label='GRIN coating')
    ax.plot(thicknesses, R_skk_arr, 's-', color=GREEN, lw=2.5, ms=7, label='sKK coating')
    ax.plot(thicknesses, R_gated_arr, '^-', color=PURPLE, lw=2.5, ms=7,
            label=r'sKK gated ($\sigma$=0.1)')
    ax.set_xlabel(r'Coating thickness ($\mu$m)')
    ax.set_ylabel(r'$\langle R_{\rm back} \rangle$')
    ax.set_title(r'Backside reflection vs coating thickness (s-pol, 80°, $\lambda$-averaged 2–5 $\mu$m)')
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
    plt.savefig(f'{FIGDIR}/fig9_thickness_single.png')
    plt.close()
    print("Saved fig 10: Thickness design space")

    # ====================================================================
    # LOSS SHAPE COMPARISON
    # Does sKK shape matter, or just total loss?
    # Hold ε'(x) fixed, vary ε''(x) shape with same ∫ε''dx, compare R_back.
    # ====================================================================

    # Use dense grid for shape comparison
    # xx_dense, e_re_dense, e_im_dense already computed above (line ~477)

    def make_constant_profile(xx, target_integral):
        """Constant ε'' everywhere, scaled to match target integral."""
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
        """Two Gaussians at ±L/4."""
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
        # Smooth with Gaussian filter
        from scipy.ndimage import gaussian_filter1d
        sigma_filt = len(xx) // 20
        smooth = gaussian_filter1d(noise, sigma_filt)
        smooth -= smooth.min()  # ensure non-negative
        integral = np.trapezoid(smooth, xx)
        if integral > 0:
            smooth *= target_integral / integral
        return smooth

    def plot_shape_figure(xx, e_re, e_im_shape, shape_name, batch_name, fname,
                          ndata, kdata, lamdata, angle_deg, pol, delta):
        """Plot a two-panel figure: profile (left) + R_back/A vs wavelength (right)."""
        # Discretize and run TMM
        ee = e_re + 1j * e_im_shape
        nc, dc = discretize_profile(xx, ee, delta=delta)
        Rb, At = Rback_vs_wavelength(nc, dc, ndata, kdata, lamdata, angle_deg, pol)

        R_avg = np.trapezoid(Rb, lamdata) / (lamdata[-1] - lamdata[0])
        A_avg = np.trapezoid(At, lamdata) / (lamdata[-1] - lamdata[0])
        sf = spectral_fom(xx, e_re, e_im_shape)
        total_loss = np.trapezoid(e_im_shape, xx)

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

        # Left panel: ε'(x) and ε''(x)
        ax_l2 = ax_l.twinx()
        ax_l.plot(xx, e_re, color=BLUE, lw=2.5)
        ax_l2.plot(xx, e_im_shape, color=RED, lw=2.5)
        ax_l.set_xlabel(r'$x$ ($\mu$m)')
        ax_l.set_ylabel(r"$\epsilon'$", color=BLUE)
        ax_l2.set_ylabel(r"$\epsilon''$", color=RED)
        ax_l.tick_params(axis='y', labelcolor=BLUE)
        ax_l2.tick_params(axis='y', labelcolor=RED)
        ax_l.set_title(f'{shape_name}')
        # Add top padding to right y-axis so textbox sits above all curves
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
        plt.savefig(f'{FIGDIR}/{fname}')
        plt.close()
        return R_avg, A_avg, sf, total_loss

    # --- Batch 1: Unconstrained loss placement ---
    print("\n=== LOSS SHAPE COMPARISON — Batch 1: Unconstrained ===")
    loss_ref = np.trapezoid(e_im_dense, xx_dense)
    print(f"  Reference total loss (sKK): ∫ε''dx = {loss_ref:.6f}")

    shape_names = ['sKK (HT derivative)', 'Constant', 'Gaussian', 'Double peaks', 'Random']
    shape_fnames = ['skk', 'constant', 'gaussian', 'double', 'random']
    shapes_b1 = [
        np.copy(e_im_dense),
        make_constant_profile(xx_dense, loss_ref),
        make_gaussian_profile(xx_dense, loss_ref),
        make_double_peak_profile(xx_dense, loss_ref),
        make_random_profile(xx_dense, loss_ref),
    ]

    print(f"  {'Shape':<22s} {'∫ε″dx':>10s} {'⟨R⟩':>10s} {'⟨A⟩':>10s} {'FoM%':>8s}")
    print(f"  {'-'*54}")
    for name, fname, e_im_shape in zip(shape_names, shape_fnames, shapes_b1):
        R_avg, A_avg, sf, tl = plot_shape_figure(
            xx_dense, e_re_dense, e_im_shape, name, 'Batch 1 (unconstrained)',
            f'fig_batch1_{fname}.png', ndata, kdata, lamdata, angle_test, pol_test, delta)
        print(f"  {name:<22s} {tl:10.6f} {R_avg:10.5f} {A_avg:10.5f} {sf:8.1f}")
        print(f"    Saved fig_batch1_{fname}.png")

    # --- Batch 2: Gated loss placement ---
    print("\n=== LOSS SHAPE COMPARISON — Batch 2: Gated ===")
    gate_dense = smooth_gate(e_re_dense, 1.3**2, 0.1)
    e_im_gated_ref = e_im_dense * gate_dense
    loss_ref_gated = np.trapezoid(e_im_gated_ref, xx_dense)
    print(f"  Reference total loss (gated sKK): ∫ε''dx = {loss_ref_gated:.6f}")

    shapes_b2 = []
    for e_im_shape in shapes_b1:
        gated = e_im_shape * gate_dense
        gated_integral = np.trapezoid(gated, xx_dense)
        if gated_integral > 1e-15:
            gated *= loss_ref_gated / gated_integral
        shapes_b2.append(gated)

    print(f"  {'Shape':<22s} {'∫ε″dx':>10s} {'⟨R⟩':>10s} {'⟨A⟩':>10s} {'FoM%':>8s}")
    print(f"  {'-'*54}")
    for name, fname, e_im_shape in zip(shape_names, shape_fnames, shapes_b2):
        R_avg, A_avg, sf, tl = plot_shape_figure(
            xx_dense, e_re_dense, e_im_shape, name, 'Batch 2 (gated)',
            f'fig_batch2_{fname}.png', ndata, kdata, lamdata, angle_test, pol_test, delta)
        print(f"  {name:<22s} {tl:10.6f} {R_avg:10.5f} {A_avg:10.5f} {sf:8.1f}")
        print(f"    Saved fig_batch2_{fname}.png")

    # ====================================================================
    # WIDTH-AMPLITUDE TRADEOFF
    # Scale ε''(x) width by factor s while keeping ∫ε''dx constant.
    # ε''_s(x) = (1/s) · ε''(x_c + (x − x_c)/s), where x_c is peak center.
    # ====================================================================
    from scipy.interpolate import interp1d

    print("\n=== WIDTH-AMPLITUDE TRADEOFF ===")

    # Find peak center of the reference ε''
    x_center = xx_dense[np.argmax(e_im_dense)]
    print(f"  ε'' peak center: x_c = {x_center:.4f} μm")

    loss_ref_width = np.trapezoid(e_im_dense, xx_dense)

    def scale_eim(xx, eim_ref, x_c, s, target_loss):
        """Scale ε'' width by factor s around x_c, then rescale to preserve ∫ε''dx."""
        interp = interp1d(xx, eim_ref, kind='cubic', bounds_error=False, fill_value=0.0)
        x_scaled = x_c + (xx - x_c) / s
        eim_scaled = interp(x_scaled) / s
        # Rescale to exact target loss (compensates for grid clipping at large s)
        current_loss = np.trapezoid(eim_scaled, xx)
        if current_loss > 1e-15:
            eim_scaled *= target_loss / current_loss
        return eim_scaled

    # --- Figure 1: Overlay of 5 ε'' curves ---
    s_discrete = [0.5, 0.75, 1.0, 1.5, 2.0]
    colors_s = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd']

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for s_val, col in zip(s_discrete, colors_s):
        eim_s = scale_eim(xx_dense, e_im_dense, x_center, s_val, loss_ref_width)
        loss_s = np.trapezoid(eim_s, xx_dense)
        label = f'$s$ = {s_val}'
        if s_val == 1.0:
            label += ' (reference)'
        ax.plot(xx_dense, eim_s, color=col, lw=2.5 if s_val == 1.0 else 1.8, label=label)
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r"$\epsilon''(x)$")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.1, 0.1)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig_width_profiles.png')
    plt.close()
    print("  Saved fig_width_profiles.png")

    # Verify total loss is constant
    for s_val in s_discrete:
        eim_s = scale_eim(xx_dense, e_im_dense, x_center, s_val, loss_ref_width)
        loss_s = np.trapezoid(eim_s, xx_dense)
        print(f"    s={s_val:.2f}: ∫ε''dx = {loss_s:.6f}")

    # --- Figure 2: ⟨R⟩ and ⟨A⟩ vs s (continuous sweep) ---
    s_sweep = np.linspace(0.3, 3.0, 50)
    R_vs_s = np.zeros(len(s_sweep))
    A_vs_s = np.zeros(len(s_sweep))

    for j, s_val in enumerate(s_sweep):
        eim_s = scale_eim(xx_dense, e_im_dense, x_center, s_val, loss_ref_width)
        ee_s = e_re_dense + 1j * eim_s
        nc_s, dc_s = discretize_profile(xx_dense, ee_s, delta=delta)
        Rb_s, At_s = Rback_vs_wavelength(nc_s, dc_s, ndata, kdata, lamdata, angle_test, pol_test)
        R_vs_s[j] = np.trapezoid(Rb_s, lamdata) / (lamdata[-1] - lamdata[0])
        A_vs_s[j] = np.trapezoid(At_s, lamdata) / (lamdata[-1] - lamdata[0])
        if (j + 1) % 10 == 0:
            print(f"    s sweep: {j+1}/{len(s_sweep)}")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(s_sweep, R_vs_s, color=GREEN, lw=2.5, label=r'$\langle R_{\rm back} \rangle_\lambda$')
    ax.plot(s_sweep, A_vs_s, color=RED, lw=2.5, label=r'$\langle A \rangle_\lambda$')

    # Mark discrete s values
    for s_val, col in zip(s_discrete, colors_s):
        idx = np.argmin(np.abs(s_sweep - s_val))
        ax.plot(s_val, R_vs_s[idx], 'o', color=col, ms=9, zorder=5,
                markeredgecolor='black', markeredgewidth=1.0)
        ax.plot(s_val, A_vs_s[idx], 's', color=col, ms=9, zorder=5,
                markeredgecolor='black', markeredgewidth=1.0)

    ax.set_xlabel(r'Width scaling factor $s$', fontsize=14)
    ax.set_ylabel('Fraction of Power', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig_width_sweep.png')
    plt.close()
    print("  Saved fig_width_sweep.png")

    # Print values at discrete s
    print(f"\n  {'s':>5s}  {'⟨R⟩':>10s}  {'⟨A⟩':>10s}")
    print(f"  {'-'*30}")
    for s_val in s_discrete:
        idx = np.argmin(np.abs(s_sweep - s_val))
        print(f"  {s_val:5.2f}  {R_vs_s[idx]:10.5f}  {A_vs_s[idx]:10.5f}")

    # --- Figure 3: Equation render ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 2.5))
    ax.axis('off')
    line1 = r"$\epsilon''_s(x) = \frac{1}{s}\;\epsilon''\!\left(x_c + \frac{x - x_c}{s}\right)$"
    line2 = r"$\int \epsilon''_s\, dx \;=\; \int \epsilon''\, dx \quad \forall\; s$"
    ax.text(0.5, 0.65, line1, transform=ax.transAxes, fontsize=24,
            ha='center', va='center')
    ax.text(0.5, 0.25, line2, transform=ax.transAxes, fontsize=24,
            ha='center', va='center')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig_width_equation.png')
    plt.close()
    print("  Saved fig_width_equation.png")

    # ====================================================================
    # THICKER COATING — LOSS SHAPE COMPARISON (Batch 1 only)
    # k_steep=10 → thickness ≈ 4 μm, comparable to λ = 2–5 μm.
    # Hypothesis: sKK shape should matter more when coating is resolvable.
    # ====================================================================
    print("\n=== LOSS SHAPE COMPARISON — Thicker Coating (k_steep=10) ===")

    k_steep_thick = 10
    dx_thick = 1 / (100 * k_steep_thick)
    xmin_thick = -20 / k_steep_thick
    xmax_thick = -xmin_thick
    xx_thick = np.linspace(xmin_thick, xmax_thick,
                           1 + int(np.floor((xmax_thick - xmin_thick) / dx_thick)))
    e_re_thick = logistic_eps(xx_thick, k_steep_thick, nb)
    e_im_thick = ht_derivative(xx_thick, e_re_thick)

    print(f"  k_steep = {k_steep_thick}, grid: {xmin_thick} to {xmax_thick} μm, "
          f"{len(xx_thick)} pts")
    print(f"  Coating thickness ≈ {xmax_thick - xmin_thick:.1f} μm")

    loss_ref_thick = np.trapezoid(e_im_thick, xx_thick)
    print(f"  Reference total loss (sKK): ∫ε''dx = {loss_ref_thick:.6f}")

    shapes_thick = [
        np.copy(e_im_thick),
        make_constant_profile(xx_thick, loss_ref_thick),
        make_gaussian_profile(xx_thick, loss_ref_thick),
        make_double_peak_profile(xx_thick, loss_ref_thick),
        make_random_profile(xx_thick, loss_ref_thick),
    ]

    print(f"  {'Shape':<22s} {'∫ε″dx':>10s} {'⟨R⟩':>10s} {'⟨A⟩':>10s} {'FoM%':>8s}")
    print(f"  {'-'*54}")
    for name, fname, e_im_shape in zip(shape_names, shape_fnames, shapes_thick):
        R_avg, A_avg, sf, tl = plot_shape_figure(
            xx_thick, e_re_thick, e_im_shape, name, 'Thick coating',
            f'fig_batch1_thick_{fname}.png', ndata, kdata, lamdata,
            angle_test, pol_test, delta)
        print(f"  {name:<22s} {tl:10.6f} {R_avg:10.5f} {A_avg:10.5f} {sf:8.1f}")
        print(f"    Saved fig_batch1_thick_{fname}.png")

    # ====================================================================
    # WIDTH-AMPLITUDE TRADEOFF — THICK COATING (k_steep=10, ~4 μm)
    # Same analysis as above but for wavelength-scale coating where
    # we expect actual s-dependence (unlike sub-λ thin coating).
    # ====================================================================
    print("\n=== WIDTH-AMPLITUDE TRADEOFF — Thick Coating (k_steep=10) ===")

    # Find peak center of the thick ε''
    x_center_thick = xx_thick[np.argmax(e_im_thick)]
    print(f"  ε'' peak center: x_c = {x_center_thick:.4f} μm")

    loss_ref_width_thick = np.trapezoid(e_im_thick, xx_thick)

    # --- Figure 1: Overlay of 5 ε'' curves (broader discrete values) ---
    s_discrete_thick = [0.25, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0]
    colors_s_thick = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for s_val, col in zip(s_discrete_thick, colors_s_thick):
        eim_s = scale_eim(xx_thick, e_im_thick, x_center_thick, s_val, loss_ref_width_thick)
        label = f'$s$ = {s_val}'
        if s_val == 1.0:
            label += ' (reference)'
        ax.plot(xx_thick, eim_s, color=col, lw=2.5 if s_val == 1.0 else 1.8, label=label)
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r"$\epsilon''(x)$")
    ax.legend(fontsize=10)
    ax.set_xlim(-1.5, 1.5)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig_width_thick_profiles.png')
    plt.close()
    print("  Saved fig_width_thick_profiles.png")

    # Verify total loss is constant
    for s_val in s_discrete_thick:
        eim_s = scale_eim(xx_thick, e_im_thick, x_center_thick, s_val, loss_ref_width_thick)
        loss_s = np.trapezoid(eim_s, xx_thick)
        print(f"    s={s_val:.2f}: ∫ε''dx = {loss_s:.6f}")

    # --- Figure 2: ⟨R⟩ and ⟨A⟩ vs s (continuous sweep, broader range) ---
    s_sweep_thick = np.linspace(0.1, 10.0, 80)
    R_vs_s_thick = np.zeros(len(s_sweep_thick))
    A_vs_s_thick = np.zeros(len(s_sweep_thick))

    for j, s_val in enumerate(s_sweep_thick):
        eim_s = scale_eim(xx_thick, e_im_thick, x_center_thick, s_val, loss_ref_width_thick)
        ee_s = e_re_thick + 1j * eim_s
        nc_s, dc_s = discretize_profile(xx_thick, ee_s, delta=delta)
        Rb_s, At_s = Rback_vs_wavelength(nc_s, dc_s, ndata, kdata, lamdata, angle_test, pol_test)
        R_vs_s_thick[j] = np.trapezoid(Rb_s, lamdata) / (lamdata[-1] - lamdata[0])
        A_vs_s_thick[j] = np.trapezoid(At_s, lamdata) / (lamdata[-1] - lamdata[0])
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
    plt.savefig(f'{FIGDIR}/fig_width_thick_sweep.png')
    plt.close()
    print("  Saved fig_width_thick_sweep.png")

    # Print values at discrete s
    print(f"\n  {'s':>5s}  {'⟨R⟩':>10s}  {'⟨A⟩':>10s}")
    print(f"  {'-'*30}")
    for s_val in s_discrete_thick:
        idx = np.argmin(np.abs(s_sweep_thick - s_val))
        print(f"  {s_val:5.2f}  {R_vs_s_thick[idx]:10.5f}  {A_vs_s_thick[idx]:10.5f}")

    print("\n=== ALL FIGURES GENERATED ===")
