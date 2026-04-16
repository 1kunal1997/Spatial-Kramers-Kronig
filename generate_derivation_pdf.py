"""
generate_derivation_pdf.py
==========================
Produces derivation_fourier_spectra.pdf from scratch using matplotlib's
mathtext engine.  Run with:
    conda run -n env python generate_derivation_pdf.py
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# ─── global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 10.5,
})

PAGE_W, PAGE_H = 8.5, 11.0     # inches
MARGIN_L, MARGIN_R = 0.95, 0.95
MARGIN_T, MARGIN_B = 0.70, 0.55
TEXT_W = PAGE_W - MARGIN_L - MARGIN_R
TEXT_H = PAGE_H - MARGIN_T - MARGIN_B

BLUE = '#1f77b4'

# ─── helper: write a page of text lines ──────────────────────────────────────

def new_page(pdf, title=None):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.axis('off')
    if title:
        ax.text(PAGE_W / 2, PAGE_H - 0.38, title, ha='center', va='top',
                fontsize=15, fontweight='bold', fontfamily='serif')
    return fig, ax


def close_page(fig, pdf):
    pdf.savefig(fig)
    plt.close(fig)


def text(ax, x, y, s, **kw):
    """Thin wrapper; default left-aligned."""
    kw.setdefault('ha', 'left')
    kw.setdefault('va', 'top')
    kw.setdefault('wrap', False)
    kw.setdefault('fontsize', 10.5)
    ax.text(x, y, s, **kw)
    return ax


def hline(ax, y, lw=0.7):
    ax.axhline(y, xmin=MARGIN_L / PAGE_W,
               xmax=(PAGE_W - MARGIN_R) / PAGE_W,
               color='black', lw=lw)


# ─── content lines ────────────────────────────────────────────────────────────
# Each item is a (kind, content) tuple.
# Kinds: 'h1','h2','h3', 'body', 'eq', 'vsp', 'hline', 'table'

LOC = 0   # mutable cursor (y position)
DY_BODY = 0.195
DY_EQ   = 0.265
DY_H1   = 0.36
DY_H2   = 0.30
DY_H3   = 0.24
DY_VS   = 0.10

SECTIONS = []   # list of pages, each page = list of (kind, text, extra_dy)


def p(kind, content, dy=None):
    """Append an item to the last page."""
    SECTIONS[-1].append((kind, content, dy))


def new_section():
    SECTIONS.append([])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1  – Title + Overview + FT convention
# ══════════════════════════════════════════════════════════════════════════════
new_section()
p('title',
  r'Analytical Fourier Power Spectra of the', None)
p('title2',
  r'Lorentzian and Logistic Dielectric Profiles', None)
p('vsp', '', 0.25)

p('body',
  r'These notes derive the analytical forms of $|\hat{\varepsilon}(k)|^2$ for two', None)
p('body',
  r'profiles used in the spatial Kramers–Kronig (sKK) framework:', None)
p('body',
  r'   1.  A Lorentzian complex profile $\varepsilon = \varepsilon^\prime + i\varepsilon^{\prime\prime}$'
  r'  analytic in the UHP.', None)
p('body',
  r'   2.  A logistic profile with $\varepsilon^{\prime\prime} = \mathcal{H}[\varepsilon^\prime]$.', None)
p('vsp', '', 0.12)

p('hline', '', None)
p('h2', r'Fourier transform convention', None)
p('body',
  r'Throughout, $k$ is the angular spatial frequency (rad $\mu$m$^{-1}$):', None)
p('eq',
  r'$\hat{\varepsilon}(k)=\displaystyle\int_{-\infty}^{\infty}\varepsilon(x)\,e^{-ikx}\,dx'
  r'\qquad\longleftrightarrow\qquad'
  r'\varepsilon(x)=\frac{1}{2\pi}\int_{-\infty}^{\infty}\hat{\varepsilon}(k)\,e^{ikx}\,dk$', None)
p('body',
  r'This matches NumPy\'s FFT convention: $k = 2\pi\,\mathrm{fftfreq}(\ldots)$.'
  r'  Values on figure x-axes are angular frequency in rad $\mu$m$^{-1}$;', None)
p('body',
  r'the label "$\mu\mathrm{m}^{-1}$" is standard (rad is dimensionless).', None)

p('vsp', '', 0.18)
p('hline', '', None)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Lorentzian: profile + contour setup
# ══════════════════════════════════════════════════════════════════════════════
new_section()
p('h1', r'1.  The Lorentzian Complex Profile', None)

p('h2', r'1.1  Profile definition', None)
p('body',
  r'The Lorentzian dielectric profile is', None)
p('eq',
  r'$\varepsilon(x) = n_b^2 - \dfrac{a\gamma}{x + i\gamma},$'
  r'\qquad a,\,\gamma > 0,$', None)
p('body',
  r'with real and imaginary parts', None)
p('eq',
  r'$\varepsilon^\prime(x) = n_b^2 - \dfrac{a\gamma\, x}{x^2+\gamma^2},'
  r'\qquad'
  r'\varepsilon^{\prime\prime}(x) = \dfrac{a\gamma^2}{x^2+\gamma^2}.$', None)
p('body',
  r'The function $\varepsilon(z)$ is analytic in the upper half $z$-plane (UHP);'
  r' its only pole lies at $z_0 = -i\gamma$, strictly in the lower half-plane (LHP).', None)

p('vsp', '', 0.12)
p('h2', r'1.2  Fourier transform via contour integration', None)
p('body',
  r'The constant $n_b^2$ contributes only a $\delta(k)$ term and is excluded ($k\neq 0$).'
  r'  Writing the FT of the non-trivial part:', None)
p('eq',
  r'$\hat{\varepsilon}(k) = -a\gamma\displaystyle\int_{-\infty}^{\infty}'
  r'\dfrac{e^{-ikx}}{x+i\gamma}\,dx, \qquad k\neq 0.$', None)

p('body',
  r'Promote $x\!\to\!z = x+iy\in\mathbb{C}$ and append a large semicircular arc of radius $R\to\infty$.'
  r'  On the arc $z = Re^{i\theta}$:', None)
p('eq',
  r'$|e^{-ikz}| = e^{\,k\,\mathrm{Im}(-iz)} = e^{-kR\sin\theta}.$', None)
p('body',
  r'Jordan\'s lemma guarantees the arc integral vanishes:', None)
p('body',
  r'   $\bullet$  Close in the \textbf{LHP} ($\sin\theta < 0$)'
  r' $\Rightarrow$ $|e^{-ikz}|\to 0$ for $k>0$.',
  None)
p('body',
  r'   $\bullet$  Close in the \textbf{UHP} ($\sin\theta > 0$)'
  r' $\Rightarrow$ $|e^{-ikz}|\to 0$ for $k<0$.',
  None)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Lorentzian: k>0 and k<0, result
# ══════════════════════════════════════════════════════════════════════════════
new_section()
p('h2', r'1.3  Case $k > 0$ (close in the LHP, clockwise contour)', None)
p('body',
  r'The CW contour encloses $z_0 = -i\gamma$ (in the LHP).'
  r'  The residue theorem for a CW contour gives a factor $-2\pi i$:', None)
p('eq',
  r'$\hat{\varepsilon}(k) = -a\gamma\cdot(-2\pi i)\cdot'
  r'\underbrace{\operatorname{Res}_{z=-i\gamma}'
  r'\!\dfrac{e^{-ikz}}{z+i\gamma}}_{\displaystyle =\; e^{-ik(-i\gamma)}\;=\;e^{-k\gamma}}.$', None)
p('body',
  r'The residue of $e^{-ikz}/(z+i\gamma)$ at the simple pole $z_0=-i\gamma$:', None)
p('eq',
  r'$\operatorname{Res}_{z=-i\gamma}\dfrac{e^{-ikz}}{z+i\gamma}'
  r'= \lim_{z\to -i\gamma}(z+i\gamma)\cdot\dfrac{e^{-ikz}}{z+i\gamma}'
  r'= e^{-ik(-i\gamma)} = e^{-k\gamma}.$', None)
p('body', r'Therefore:', None)
p('eq',
  r'$\boxed{\hat{\varepsilon}(k) = 2\pi i\,a\gamma\,e^{-k\gamma}, \qquad k>0.}$', None)

p('vsp', '', 0.10)
p('h2', r'1.4  Case $k < 0$ (close in the UHP, counter-clockwise)', None)
p('body',
  r'The CCW contour encloses the UHP.  No poles lie in the UHP'
  r' (the only pole is $z_0=-i\gamma$ in the LHP), so by the'
  r' residue theorem:', None)
p('eq',
  r'$\hat{\varepsilon}(k) = 0, \qquad k<0.$', None)
p('body',
  r'This confirms that the Lorentzian profile is an analytic signal:'
  r' all spectral weight lies at $k>0$, consistent with $\varepsilon(z)$ being'
  r' analytic above the real axis.', None)

p('vsp', '', 0.10)
p('h2', r'1.5  Power spectrum', None)
p('body',
  r'Squaring the modulus of $\hat{\varepsilon}(k)$ for $k>0$:', None)
p('eq',
  r'$\boxed{|\hat{\varepsilon}(k)|^2 = 4\pi^2 a^2\gamma^2\,e^{-2\gamma k}, \qquad k>0.}$', None)
p('body',
  r'The exponential decay rate $2\gamma$ equals twice the imaginary distance'
  r' from the real axis to the nearest pole.  The prefactor $4\pi^2 a^2\gamma^2$'
  r' is exact — no free parameters.', None)

p('vsp', '', 0.10)
p('body',
  r'Numerical implementation (from \texttt{fig\_lorentzian\_crossover\_M2000.py}):', None)
p('body',
  r'     $C = 4\pi^2 a^2\gamma^2,$  $|\hat{\varepsilon}(k)|^2 = C\,e^{-2\gamma k}$.'
  r'  Parameters: $a=1,\;\gamma=0.01\;\mu\mathrm{m},\;n_b=1.5$.',
  None)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Logistic: profile + analytic signal
# ══════════════════════════════════════════════════════════════════════════════
new_section()
p('h1', r'2.  The Logistic Profile', None)

p('h2', r'2.1  Profile definition', None)
p('body',
  r'The logistic dielectric profile is real-valued:', None)
p('eq',
  r'$\varepsilon^\prime(x) = 1 + \dfrac{\Delta}{1+e^{k_s x}},'
  r'\qquad \Delta = n_b^2 - 1,\; k_s > 0,$', None)
p('body',
  r'interpolating from $\varepsilon^\prime(-\infty) = n_b^2$ to'
  r' $\varepsilon^\prime(+\infty) = 1$ over width $1/k_s$.'
  r'  Because $\varepsilon^\prime(x)$ is real and has poles in'
  r' \emph{both} half-planes, it is not by itself a valid sKK profile.'
  r'  We define the imaginary part $\varepsilon^{\prime\prime} = \mathcal{H}[\varepsilon^\prime]$.', None)

p('vsp', '', 0.13)
p('h2', r'2.2  Analytic signal: simplifying $\hat{\varepsilon}(k)$', None)
p('body',
  r'For any real function $f(x)$, the Hilbert transform $g = \mathcal{H}[f]$'
  r' satisfies in Fourier space:', None)
p('eq',
  r'$\hat{g}(k) = -i\,\mathrm{sgn}(k)\,\hat{f}(k).$', None)
p('body',
  r'Form the analytic signal $\varepsilon = \varepsilon^\prime + i\varepsilon^{\prime\prime}$'
  r' with $\varepsilon^{\prime\prime}=\mathcal{H}[\varepsilon^\prime]$.  Its FT:', None)
p('eq',
  r'$\hat{\varepsilon}(k)'
  r'= \hat{\varepsilon}^\prime(k) + i\,\hat{\varepsilon}^{\prime\prime}(k)'
  r'= \hat{\varepsilon}^\prime(k) + i\,\bigl(-i\,\mathrm{sgn}(k)\bigr)\hat{\varepsilon}^\prime(k)'
  r'= \hat{\varepsilon}^\prime(k)\bigl(1+\mathrm{sgn}(k)\bigr).$', None)
p('body',
  r'Evaluating $(1+\mathrm{sgn}(k))$:', None)
p('eq',
  r'$\hat{\varepsilon}(k) = \begin{cases}'
  r'2\,\hat{\varepsilon}^\prime(k), & k>0, \\'
  r'0, & k<0.'
  r'\end{cases}$', None)
p('body',
  r'\textbf{Key point}: the $k<0$ sector is identically zero regardless of'
  r' where the poles of $\varepsilon^\prime(z)$ lie in the complex plane.'
  r'  The sKK one-sidedness is enforced by the \emph{construction}'
  r' $\varepsilon^{\prime\prime}=\mathcal{H}[\varepsilon^\prime]$.', None)
p('body',
  r'We therefore need only compute $\hat{\varepsilon}^\prime(k)$ for $k>0$.', None)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – Logistic: contour integral + residue sum
# ══════════════════════════════════════════════════════════════════════════════
new_section()
p('h2', r'2.3  Computing $\hat{\varepsilon}^\prime(k)$ for $k>0$', None)
p('body',
  r'Discarding the constant (affects only $k=0$):', None)
p('eq',
  r'$\hat{\varepsilon}^\prime(k) = \Delta\displaystyle\int_{-\infty}^{\infty}'
  r'\dfrac{e^{-ikx}}{1+e^{k_s x}}\,dx,\qquad k>0.$', None)

p('h3', r'Poles of the integrand', None)
p('body',
  r'$1+e^{k_s z}=0 \;\Rightarrow\; e^{k_s z}=-1 = e^{i\pi(2m-1)}$'
  r' for $m\in\mathbb{Z}$, giving', None)
p('eq',
  r'$z_m = \dfrac{i\pi(2m-1)}{k_s}, \qquad m\in\mathbb{Z}.$', None)
p('body',
  r'All poles lie on the imaginary axis.  For $k>0$ close in the LHP'
  r' ($\mathrm{Im}(z)<0$), enclosing the poles with $2m-1<0$, i.e.\ $m\leq 0$.  '
  r'Re-labelling $n=1-m\geq 1$:', None)
p('eq',
  r'$z_n = -\dfrac{i\pi(2n-1)}{k_s}, \qquad n=1,2,3,\ldots$', None)
p('body',
  r'The nearest LHP pole is $z_1 = -i\pi/k_s$, at distance $\pi/k_s$ from'
  r' the real axis.  The factor $\pi$ arises from $\ln(-1)=i\pi$,'
  r' not $\ln(1)=0$.', None)

p('h3', r'Residues', None)
p('body',
  r'Using the residue formula for $f(z)/g(z)$ at a simple zero $z_0$ of $g$:'
  r' $\operatorname{Res} = f(z_0)/g^\prime(z_0)$.  With'
  r' $f=\Delta e^{-ikz}$, $g=1+e^{k_s z}$, $g^\prime=k_s e^{k_s z}$:', None)
p('eq',
  r'$\operatorname{Res}_{z=z_n}\dfrac{\Delta e^{-ikz}}{1+e^{k_s z}}'
  r'= \dfrac{\Delta e^{-ikz_n}}{k_s\,e^{k_s z_n}}'
  r'= \dfrac{\Delta e^{-ikz_n}}{k_s\cdot(-1)}'
  r'= -\dfrac{\Delta}{k_s}\,e^{-ikz_n},$', None)
p('body',
  r'where we used $e^{k_s z_n}=-1$.  The exponent evaluates to:', None)
p('eq',
  r'$-ikz_n = -ik\!\cdot\!\left(-\dfrac{i\pi(2n-1)}{k_s}\right)'
  r'= -\dfrac{\pi k(2n-1)}{k_s}.$', None)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – Logistic: geometric series + final result + asymptotes
# ══════════════════════════════════════════════════════════════════════════════
new_section()
p('h2', r'2.4  Summing the residues (CW contour)', None)
p('eq',
  r'$\hat{\varepsilon}^\prime(k)'
  r'= (-2\pi i)\sum_{n=1}^{\infty}\left(-\dfrac{\Delta}{k_s}\right)'
  r'e^{-\pi k(2n-1)/k_s}'
  r'= \dfrac{2\pi i\,\Delta}{k_s}\sum_{n=1}^{\infty}e^{-\pi k(2n-1)/k_s}.$', None)

p('h3', r'Evaluating the geometric series', None)
p('body',
  r'Set $\beta\equiv\pi k/k_s > 0$:', None)
p('eq',
  r'$\displaystyle\sum_{n=1}^{\infty}e^{-\beta(2n-1)}'
  r'= e^{-\beta}+e^{-3\beta}+e^{-5\beta}+\cdots'
  r'= \dfrac{e^{-\beta}}{1-e^{-2\beta}}'
  r'= \dfrac{1}{e^{\beta}-e^{-\beta}}'
  r'= \dfrac{1}{2\sinh\beta}.$', None)
p('body', r'Therefore:', None)
p('eq',
  r'$\hat{\varepsilon}^\prime(k)'
  r'= \dfrac{2\pi i\,\Delta}{k_s}\cdot\dfrac{1}{2\sinh(\pi k/k_s)}'
  r'= \dfrac{\pi i\,\Delta}{k_s\sinh(\pi k/k_s)}.$', None)
p('body',
  r'From the analytic-signal result $\hat{\varepsilon}(k)=2\hat{\varepsilon}^\prime(k)$:', None)
p('eq',
  r'$\hat{\varepsilon}(k) = \dfrac{2\pi i\,\Delta}{k_s\sinh(\pi k/k_s)},\qquad k>0.$', None)

p('vsp', '', 0.08)
p('h2', r'2.5  Power spectrum', None)
p('eq',
  r'$\boxed{|\hat{\varepsilon}(k)|^2 = \dfrac{C}{\sinh^2(\pi k/k_s)},'
  r'\quad C=\dfrac{4\pi^2\Delta^2}{k_s^2},\quad k>0.}$', None)
p('body',
  r'No free parameters; both the prefactor $C$ and the shape'
  r' $1/\sinh^2(\cdot)$ are set by the profile parameters.', None)

p('vsp', '', 0.10)
p('h2', r'2.6  Asymptotic regimes', None)

p('h3', r'Small $k$ (algebraic regime, $k\ll k_s/\pi$)', None)
p('body', r'Use the small-argument approximation $\sinh(x)\approx x$:', None)
p('eq',
  r'$|\hat{\varepsilon}(k)|^2 \approx'
  r'\dfrac{4\pi^2\Delta^2/k_s^2}{(\pi k/k_s)^2}'
  r'= \dfrac{4\Delta^2}{k^2}.$', None)
p('body',
  r'This $1/k^2$ power law is characteristic of a profile with a'
  r' finite step discontinuity of height $\Delta$.', None)

p('h3', r'Large $k$ (exponential regime, $k\gg k_s/\pi$)', None)
p('body', r'Use $\sinh(x)\approx e^x/2$ for $x\gg 1$:', None)
p('eq',
  r'$|\hat{\varepsilon}(k)|^2 \approx'
  r'\dfrac{C}{\tfrac{1}{4}e^{2\pi k/k_s}}'
  r'= 4C\,e^{-2\pi k/k_s}.$', None)
p('body',
  r'Decay rate $2\pi/k_s$ is set by the nearest LHP pole at'
  r' $\mathrm{Im}(z_1)=-\pi/k_s$.', None)

p('h3', r'Crossover', None)
p('body',
  r'The two asymptotes cross when $4\Delta^2/k^2 = 4C\,e^{-2\pi k/k_s}$,'
  r' which to leading order gives', None)
p('eq', r'$k_\times \approx k_s/\pi.$', None)
p('body',
  r'For $k_s=100\;\mu\mathrm{m}^{-1}$: $k_\times\approx 32\;\mu\mathrm{m}^{-1}$.', None)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 – Comparison table
# ══════════════════════════════════════════════════════════════════════════════
new_section()
p('h1', r'3.  Comparison: Lorentzian vs.\ Logistic', None)
p('body',
  r'If the transition widths are matched via $\gamma = 1/k_s$, the large-$k$'
  r' slopes on a log-scale differ by $\pi$:', None)
p('eq',
  r'$\text{Lorentzian: slope} = -2\gamma = -2/k_s;'
  r'\qquad'
  r'\text{Logistic: slope} = -2\pi/k_s.$', None)
p('body',
  r'The factor of $\pi$ reflects the pole geometry: the Lorentzian pole sits'
  r' at imaginary distance $\gamma = 1/k_s$ from the real axis, while the'
  r' nearest logistic pole sits at $\pi/k_s$ — a direct consequence of'
  r' $\ln(-1)=i\pi$ (vs.\ $\ln e^{-1}=-1$ for the Lorentzian).', None)
p('vsp', '', 0.25)

# ─── render pages ────────────────────────────────────────────────────────────

OUTPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'derivation_fourier_spectra.pdf')

KIND_DY = {
    'title':  0.42,
    'title2': 0.42,
    'h1':     DY_H1,
    'h2':     DY_H2,
    'h3':     DY_H3,
    'body':   DY_BODY,
    'eq':     DY_EQ,
    'vsp':    None,  # use extra_dy directly
    'hline':  0.18,
}

KIND_FS = {
    'title':  15,
    'title2': 15,
    'h1':     13,
    'h2':     11.5,
    'h3':     10.8,
    'body':   10.5,
    'eq':     10.5,
    'vsp':    10.5,
    'hline':  10.5,
}

KIND_FW = {
    'title': 'bold', 'title2': 'bold',
    'h1': 'bold', 'h2': 'bold', 'h3': 'bold',
}

def render_page_number(ax, n):
    ax.text(PAGE_W / 2, MARGIN_B / 2, str(n),
            ha='center', va='center', fontsize=9, color='gray')

with PdfPages(OUTPATH) as pdf:
    for page_idx, page_items in enumerate(SECTIONS):
        fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, PAGE_W)
        ax.set_ylim(0, PAGE_H)
        ax.axis('off')

        cursor = PAGE_H - MARGIN_T
        x_body = MARGIN_L
        x_eq   = MARGIN_L + 0.35

        for kind, content, extra_dy in page_items:
            dy_map = KIND_DY.get(kind, DY_BODY)

            if kind == 'hline':
                hline(ax, cursor - 0.04)
                cursor -= dy_map
                continue

            if kind == 'vsp':
                cursor -= (extra_dy if extra_dy else DY_VS)
                continue

            fs = KIND_FS.get(kind, 10.5)
            fw = KIND_FW.get(kind, 'normal')
            ha = 'center' if kind in ('title', 'title2') else 'left'
            x  = PAGE_W / 2 if ha == 'center' else (x_eq if kind == 'eq' else x_body)

            ax.text(x, cursor, content,
                    ha=ha, va='top',
                    fontsize=fs, fontweight=fw,
                    fontfamily='serif')

            if kind == 'vsp':
                cursor -= (extra_dy or DY_VS)
            else:
                dy = extra_dy if extra_dy else dy_map
                cursor -= dy

        render_page_number(ax, page_idx + 1)
        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved: {OUTPATH}")
