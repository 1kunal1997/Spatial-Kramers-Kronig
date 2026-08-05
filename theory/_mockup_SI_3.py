"""SI Fig 3 — does a thinner, taller Gaussian beat the sKK loss shape?

Ported from the exploratory _explore_thin_gaussian.py into skk_fig_common conventions
(deterministic inch layout, exact 4:3 panels, bold outside letters, no panel titles).

  a  the eps''(x) loss profiles: sKK against matched-total-loss Gaussians of shrinking width
  b  reflectance vs lambda for the same set, s-pol, theta_b = 20 deg

Panel a is built to the SAME recipe as the loss-shape panel of the main-text loss-shape figure
(_mockup_fig5_6_combined.py panel a): x plotted as x/L over the full domain, ticks at -L/2, 0,
L/2, x-label "x", y-label eps''(x), and no title. The profile, domain (M = 20 transition widths)
and discretization (default delta) are inherited from skk_fig_common, which is what that figure
uses too -- so the two panels are directly comparable rather than merely similar.

The sKK curve is green here (not the usual dark red) so it separates cleanly from the orange
Gaussian family, which is the comparison this figure exists to make.

The lambda axis deliberately runs down to L/50, wider than the other figures, to reach the
adiabatic regime where the widest Gaussian overtakes sKK.
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (FixedLocator, FixedFormatter, LogLocator,
                               NullFormatter)
import tmm_helper as tmm_h
import theory.skk_fig_common as C

C.apply_style()

# --- canonical sKK profile + matched total loss (same profile/domain as the main figures) ---
xx, e_re, e_im_skk = C.logistic_profile()
target = float(np.trapezoid(e_im_skk, xx))


def gaussian_profile(xx, target_integral, sigma):
    """Centred Gaussian eps'' of a given sigma, rescaled to the matched total loss."""
    g = np.exp(-xx**2 / (2 * sigma**2))
    g *= target_integral / np.trapezoid(g, xx)
    return np.maximum(g, 0.0)


sigma_default = (xx[-1] - xx[0]) / 8.0            # the width used by the main loss-shape figure

# One Gaussian is width-tuned so its PEAK eps'' equals the sKK peak; for a Gaussian rescaled to
# total loss `target`, peak = target / (sigma*sqrt(2pi)), hence sigma_match below.
peak_skk = e_im_skk.max()
sigma_match = target / (peak_skk * np.sqrt(2 * np.pi))
f_match = sigma_match / sigma_default

# (sigma factor, is-peak-matched). Labels are generated from the realised sigma/L below, so no
# internal tags ("fig5", "x0.25") reach the figure.
SIGMAS = [(1.0, False), (f_match, True), (0.25, False), (0.125, False)]
gauss_cmap = plt.cm.Oranges(np.linspace(0.45, 0.95, len(SIGMAS)))

C_SKK = C.C_R                                      # green -- stands out against the oranges

# --- build TMM stacks once (geometry nb|coating|air, default delta as in the main figures) ---
skk_stack = C.build_stack(xx, e_re + 1j * e_im_skk, n_out=1.0)
L_tot = C.coating_thickness(skk_stack[1])
gauss = []
for (f, matched), col in zip(SIGMAS, gauss_cmap):
    sig = sigma_default * f
    eim = gaussian_profile(xx, target, sig)
    label = r'Gaussian, $\sigma = %.3f\,L$' % (sig / L_tot)
    if matched:
        label += '\n(peak matched to sKK)'
    gauss.append(dict(eim=eim, stack=C.build_stack(xx, e_re + 1j * eim, n_out=1.0),
                      col=col, label=label, sigma=sig))
print("sKK peak eps'' = %.3f ; peak-matched Gaussian sigma = %.4f um (%.3f L), realised peak = %.3f"
      % (peak_skk, sigma_match, sigma_match / L_tot, gauss[1]['eim'].max()))
print('coating thickness L = %.3f um' % L_tot)

# --- spectral one-sidedness FoM, for the record (printed, not plotted) ---
fom_skk, _, _ = tmm_h.skk_spectral_fom(xx, e_re, e_im_skk)
print('\n=== spectral one-sidedness FoM ===')
print('  sKK                       : %.4f %%' % fom_skk)
for g in gauss:
    fom_g, _, _ = tmm_h.skk_spectral_fom(xx, e_re, g['eim'])
    print('  sigma = %.4f um           : %.4f %%   (peak = %.2f)'
          % (g['sigma'], fom_g, g['eim'].max()))

# --- sweep axis = lambda/L, kept at the exploratory script's wider L/50..10L bounds ---
LOL_MIN, LOL_MAX = 1 / 50.0, 10.0
lol = np.logspace(np.log10(LOL_MIN), np.log10(LOL_MAX), 70)
lams = lol * L_tot
LOL_TICKS = [1 / 50.0, 0.1, 1, 10]
LOL_LABELS = [r'$L/50$', r'$L/10$', r'$L$', r'$10L$']

POL, THETA = 's', 20

fig, axs, _ = C.panel_grid(nrows=1, ncols=2)
axa, axb = axs[0]

# --- panel a: loss profiles (recipe copied from the main loss-shape panel) ---
for g in gauss:
    axa.plot(xx / L_tot, g['eim'], color=g['col'], lw=C.LW_MAIN, label=g['label'])
axa.plot(xx / L_tot, e_im_skk, color=C_SKK, lw=C.LW_MAIN, label='sKK', zorder=10)
axa.set_xlim(xx.min() / L_tot, xx.max() / L_tot)
axa.xaxis.set_major_locator(FixedLocator([-0.5, 0, 0.5]))
axa.xaxis.set_major_formatter(FixedFormatter([r'$-L/2$', r'$0$', r'$L/2$']))
axa.set_xlabel(r'$x$')
axa.set_ylabel(r"$\epsilon''(x)$")

# --- panel b: reflectance vs lambda (no legend here -- panel a carries it) ---
_, Rskk, _ = C.sweep_tra_vs_wavelength(*skk_stack, lams, angle_deg=THETA, pol=POL)
for g in gauss:
    _, Rg, _ = C.sweep_tra_vs_wavelength(*g['stack'], lams, angle_deg=THETA, pol=POL)
    g['R'] = Rg
    axb.plot(lol, Rg, 'o-', color=g['col'], lw=C.LW_MAIN, ms=3.0, mew=0)
axb.plot(lol, Rskk, 's-', color=C_SKK, lw=C.LW_MAIN, ms=3.2, mew=0, zorder=10)
axb.set_xscale('log'); axb.set_yscale('log'); axb.set_xlim(LOL_MIN, LOL_MAX)
axb.xaxis.set_major_locator(FixedLocator(LOL_TICKS))
axb.xaxis.set_major_formatter(FixedFormatter(LOL_LABELS))
axb.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
axb.xaxis.set_minor_formatter(NullFormatter())
axb.set_xlabel(r'$\lambda$')
axb.set_ylabel('Reflectance')

# single-column legend INSIDE panel a, sKK first
h, l = axa.get_legend_handles_labels()
order = [len(h) - 1] + list(range(len(h) - 1))
axa.legend([h[i] for i in order], [l[i] for i in order],
           loc='upper left', ncol=1, frameon=False, fontsize=5,
           handlelength=1.4, handletextpad=0.5, labelspacing=0.35, borderpad=0.2)

C.relayout_grid(fig, axs, letters=['a', 'b'], buffer_in=0.05, col_ws_in=0.2)

C.save(fig, os.path.join('theory', '_mockup_SI_3.png'))

# --- quantitative verdict in the band the sKK shape is claimed to own ---
band = (lol >= 0.5) & (lol <= 2.0)
print('\n=== min reflectance for lambda in [L/2, 2L]  (%s-pol, %d deg) ===' % (POL, THETA))
print('  sKK              : %.3e' % Rskk[band].min())
for g in gauss:
    print('  sigma = %.4f um : %.3e   (peak eps\'\' = %.2f)'
          % (g['sigma'], g['R'][band].min(), g['eim'].max()))
