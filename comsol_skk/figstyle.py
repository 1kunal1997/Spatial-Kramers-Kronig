"""Shared matplotlib style for every figure in the sKK-as-PML study.

Imports matplotlib ONLY -- no COMSOL, no licence -- so figure scripts run anywhere.
Lifted from the old make_figures.py so new figures are visually identical to the
figures already in the Sep 2026 deck.

Usage:
    from figstyle import newax, save, C, LBL, STY, PC
    fig, ax = newax()
    ax.semilogy(theta, R, color=C['logistic'], label=LBL['logistic'])
    save(fig, 'angle_vs_terminator.png', 'one-line provenance note')
"""
import os, textwrap
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'figures')

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.28,
                     'axes.labelsize': 12, 'legend.framealpha': 0.92, 'figure.dpi': 110})

# ---------------------------------------------------------------- palette
# project convention: blue = eps', dark red = eps'', orange = derivative quantities
C = {'logistic': '#1f77b4', 'lorentzian': '#8B0000', 'tapered': '#ff7f0e',
     'turned_logistic': '#2ca02c', 'flat': '0.5',
     'eps_re': '#1f77b4', 'eps_im': '#8B0000', 'deriv': '#ff7f0e'}
LBL = {'logistic': 'logistic (DiHiTI)', 'lorentzian': 'Lorentzian',
       'tapered': 'tapered Lorentzian', 'turned_logistic': 'turned logistic'}
STY = {'logistic': '-', 'lorentzian': '-', 'tapered': '--', 'turned_logistic': '-'}
PC  = {1: 'k', 2: '0.35', 4: '0.6', 10: '0.7', 20: '0.75', 50: '0.8'}   # by PMLfactor

XL_ANGLE = r'$\theta$ (deg)'
YL_R     = r'Reflectance  $|S_{11}|^2$'


def newax(xl=XL_ANGLE, yl=YL_R, fs=(7.2, 5.2)):
    fig, ax = plt.subplots(figsize=fs)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    return fig, ax


def save(fig, name, note='', figdir=FIGDIR):
    """Write the figure with its provenance note in the grey footer."""
    os.makedirs(figdir, exist_ok=True)
    if note:
        fig.text(0.005, -0.015, '\n'.join(textwrap.wrap(note, 105)),
                 fontsize=6.8, color='0.45', va='top')
    path = os.path.join(figdir, name)
    fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print('  ', name)
    return path
