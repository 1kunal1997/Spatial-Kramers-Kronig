import tmm

from numpy import pi, inf 
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np

from plot_functions import plot_setup, plot, legend
import colors
from scipy.signal import hilbert
from scipy.signal.windows import tukey
from scipy.integrate import cumulative_trapezoid

# generate nk for spatial KK stack
def eps(x, a, gam, nb):
    return nb**2 - a * gam / (x + 1j*gam)

def logistic(x, k, nb, sx):
    return (nb**2-1) / (1 + np.exp(sx*k*x)) + 1

# %%

def hilbert_fom_derivative(x, u, v, sign=+1):
    """
    x : uniform grid
    u : real profile
    v : actual imaginary profile
    sign : choose +1 or -1 depending on your spatial-KK convention
    """

    x = np.asarray(x, float)
    u = np.asarray(u, float)
    v = np.asarray(v, float)

    # Derivative-space comparison is better for step-like profiles
    ud = np.gradient(u, x)
    vd = np.gradient(v, x)

    vd_ht = sign * np.imag(hilbert(ud))

    num = 2 * np.trapezoid(vd * vd_ht, x)
    den = np.trapezoid(vd**2, x) + np.trapezoid(vd_ht**2, x)

    fom = 100 * max(0.0, num / den)
    return fom, vd_ht

def skk_spectral_fom(x, u, v, allowed_side='positive', derivative=True):
    """
    allowed_side = 'positive' or 'negative'
    derivative=True is recommended for step-like profiles
    """
    x = np.asarray(x, float)
    u = np.asarray(u, float)
    v = np.asarray(v, float)

    if derivative:
        u = np.gradient(u, x)
        v = np.gradient(v, x)

    z = u + 1j * v

    dx = x[1] - x[0]
    Z = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(z)))
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(z), d=dx))

    if allowed_side == 'positive':
        mask_allowed = k > 0
        mask_forbidden = k < 0
    else:
        mask_allowed = k < 0
        mask_forbidden = k > 0

    E_allowed = np.sum(np.abs(Z[mask_allowed])**2)
    E_forbidden = np.sum(np.abs(Z[mask_forbidden])**2)

    fom = 100 * max(0.0, (E_allowed - E_forbidden) / (E_allowed + E_forbidden))
    return fom, k, Z

def ht_derivative(xx, e_re):
    """Derivative-then-integrate Hilbert transform method.

    Key idea: dε'/dx → 0 at both ends, so standard FFT-based HT works
    on the derivative without endpoint artifacts. Integrate back to recover ε''.
    """
    N = len(e_re)
    u = np.gradient(e_re, xx)
    v = np.imag(hilbert(u))
    e_im = cumulative_trapezoid(v, xx, initial=0)
    e_im -= np.linspace(e_im[0], e_im[-1], N)
    return e_im

def discretize_profile(xx, ee, delta=0.05):
    """Discretize continuous ε(x) into TMM layers using adaptive arc-length sampling.

    Parameters
    ----------
    xx : array — spatial coordinate
    ee : complex array — continuous dielectric function ε(x)
    delta : float — arc-length threshold for adding a new layer

    Returns
    -------
    n_list : list — refractive index of each layer
    d_list : list — thickness of each layer
    """
    e_scale = np.max(np.abs(ee - ee[0]))
    if e_scale < 1e-12:
        e_scale = 1.0
    x_scale = xx[-1] - xx[0]
    xq, eq = [xx[0]], [ee[0]]
    for k in range(1, len(xx)):
        dx = (xx[k] - xq[-1]) / x_scale
        de = abs(ee[k] - eq[-1]) / e_scale
        ds = np.sqrt(dx**2 + de**2)
        if ds > delta:
            xq.append(xx[k])
            eq.append(ee[k])
    xq.append(xx[-1])
    eq.append(ee[-1])
    xq, eq = np.array(xq), np.array(eq)
    d_list = np.diff(xq).tolist()
    e_list = (eq[:-1] + eq[1:]) / 2
    n_list = np.sqrt(e_list).tolist()
    return n_list, d_list

def _find_contiguous(xx, mask):
    """Find contiguous True regions in a boolean mask.

    Returns list of (start_x, end_x) tuples.
    """
    regions = []
    in_region = False
    for i in range(len(mask)):
        if mask[i] and not in_region:
            start = xx[i]
            in_region = True
        elif not mask[i] and in_region:
            regions.append((start, xx[i]))
            in_region = False
    if in_region:
        regions.append((start, xx[-1]))
    return regions

def plot_param_sweep(
    width_arr, y_vals, params,
    xlabel="X", ylabel="Y", paramlabel="p", title="",
    cmap_name="Blues", cmap_range=None,
    figsize=(5, 4)
):
    """
    Plot multiple curves with a smooth, self-tuned colormap gradient.

    Parameters:
        width_arr   : array-like       # Shared x-values
        y_vals      : list of arrays   # List of y-values for each curve
        params      : list or array    # Parameter values for each curve
        xlabel      : str              # X-axis label
        ylabel      : str              # Y-axis label
        paramlabel  : str              # Label prefix for legend entries
        title       : str              # Plot title
        cmap_name   : str              # Colormap name (default = 'Blues')
        cmap_range  : tuple            # Manually override colormap range, e.g. (0.2, 0.8)
        figsize     : tuple            # Figure size
    """

    params = np.array(params)
    n_curves = len(params)

    # === Auto-adjust colormap range ===
    if cmap_range is None:
        # Adaptive rule: fewer curves = broader range, more curves = slightly compressed
        if n_curves <= 5:
            start, span = 0.2, 0.8   # Balanced for 3–5 curves
        elif n_curves <= 8:
            start, span = 0.25, 0.7  # For mid-sized sets, avoid too-dark clustering
        else:
            start, span = 0.3, 0.65  # For many curves, stick to slightly lighter tones
    else:
        start, span = cmap_range

    # Create sliced colormap
    base_cmap = colormaps[cmap_name]
    cmap = lambda x: base_cmap(start + span * x)

    # Set up figure
    fig, ax = plot_setup(xlabel, ylabel, title=title,
                         xlim=(width_arr[0], width_arr[-1]),
                         figsize=figsize, auto_scale=True)

    # Plot curves
    for i, p in enumerate(params):
        color = cmap(i / (n_curves - 1)) if n_curves > 1 else cmap(0.5)
        plot(fig, ax, width_arr, y_vals[i], color=color, label=f"{paramlabel}={p}", auto_scale=True)

    # Add legend
    legend(fig, ax, auto_scale=True)

def show_textbox(text, fontsize=14):
    """
    Display a standalone textbox figure in the VS Code interactive window.

    Parameters:
        text     : str  - The content of the textbox
        fontsize : int  - Font size of the text
    """
    # Create a tiny figure
    fig, ax = plt.subplots(figsize=(2.5, 1))
    ax.axis("off")  # Hide axes

    # Add the textbox in the center
    ax.text(
        0.5, 0.5, text,
        fontsize=fontsize,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85)
    )

    # Make figure background transparent for a cleaner floating effect
    fig.patch.set_alpha(0)

def add_directional_arrows(ax, x, y, color="black", direction=1, n_arrows=5):
    """
    Draws standalone arrowheads along a solid curve.
    direction:  1  = left-to-right arrows
               -1 = right-to-left arrows
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Choose positions evenly spaced along the curve
    idxs = np.linspace(int(0.05*len(x)), int(0.95*len(x)), n_arrows, dtype=int)

    for i in idxs:

        # Calculate slope for proper arrow rotation
        dx = x[i + 1] - x[i - 1]
        dy = y[i + 1] - y[i - 1]

        # Flip arrow direction for RL curves
        dx *= direction
        dy *= direction

        ax.annotate(
            "",
            xy=(x[i], y[i]),
            xytext=(x[i] - dx * 0.001, y[i] - dy * 0.001),
            arrowprops=dict(
                arrowstyle="->",       # Just arrowheads, no tails
                color=color,
                lw=1.5,
                mutation_scale=16,     # Controls arrowhead size
            ),
        )


def plot_tra_curves(
    x, data,
    xlabel="Wavelength ($\mu$m)",
    ylabel="Fraction of Power",
    title="TRA Plot",
    xlim=None, ylim=None, figsize=(5, 4),
    auto_scale=True, add_legend=True,
    ncol_legend=1
):
    """
    Generalized TRA plotting function.

    Parameters
    ----------
    x : array
        X-axis values.
    data : dict
        Dictionary where keys are curve names and values are arrays.
        Accepted keys: T, A_RL, A_LR, R_RL, R_LR, T_bulk, A_bulk, R_bulk.
    xlabel, ylabel, title : str
        Plot labels and title.
    xlim, ylim : tuple
        Axis limits.
    figsize : tuple
        Figure size.
    auto_scale : bool
        Whether to auto-scale axes.
    add_legend : bool
        Whether to add a legend.
    text : str or None
        Optional text to display on the plot.
    text_pos : tuple
        Position for the text box in axes coordinates.
    text_box_style : dict or None
        Custom bbox style for the text box.
    """

    # Centralized config for colors, labels, styles, and arrows
    curve_config = {
        # Transmission → NEW: single key, arrowless
        "T":      {"label": r"T", "color": colors.blue, "style": "-", "arrows": 0},
        "T_bulk": {"label": r"T$_{bulk}$", "color": colors.light_blue, "style": "--", "arrows": 0},

        # Absorption
        "A_RL":   {"label": r"A$_{RL}$", "color": colors.red, "style": "-", "arrows": -1},
        "A_LR":   {"label": r"A$_{LR}$", "color": colors.red, "style": "-", "arrows": +1},
        "A_bulk": {"label": r"A$_{bulk}$", "color": colors.light_red, "style": "--", "arrows": 0},

        # Reflection
        "R_RL":   {"label": r"R$_{RL}$", "color": colors.green, "style": "-", "arrows": -1},
        "R_LR":   {"label": r"R$_{LR}$", "color": colors.green, "style": "-", "arrows": +1},
        "R_bulk": {"label": r"R$_{bulk}$", "color": colors.light_green, "style": "--", "arrows": 0},
    }

    if xlim == None:
        xlim = (x[0],x[-1])

    # --- Setup figure and axes ---
    fig, ax = plot_setup(
        xlabel, ylabel, title=title,
        xlim=xlim, ylim=ylim, figsize=figsize,
        auto_scale=auto_scale
    )

    # --- Plot each requested curve ---
    for key, y in data.items():
        if key not in curve_config:
            raise ValueError(f"Unknown curve key '{key}'. Check curve_config mapping.")

        cfg = curve_config[key]
        label, color, style, arrows = cfg["label"], cfg["color"], cfg["style"], cfg["arrows"]

        # Use your existing helper
        plot(
            fig, ax, x, y, style,
            label=label, color=color,
            auto_scale=auto_scale
        )

        # Add slope-following arrows if required
        if arrows != 0:
            add_directional_arrows(ax, x, y, color=color, direction=arrows, n_arrows=7)

    # --- Legend ---
    if add_legend:
        legend(fig, ax, auto_scale=auto_scale, ncol=ncol_legend)


# plotting function for refractive index
def nk_plot(xx, ee, nk, xq, n_list, gam, a, nb, zoomed):
    xlabel = 'x ($\mu$m)'; ylabel = 'Re(n)'
    title = f''
    if (zoomed):
        xlim = (xx[0]/10,xx[-1]/10)
    else:
        xlim = (xx[0],xx[-1])
    fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True, xlim=xlim)
    n_real = np.real(n_list)
    midpoints = (xq[:-1] + xq[1:]) / 2
    plot(fig,ax, xx, np.real(nk), label='smooth', color=colors.red,auto_scale=True)
    ax.stairs(n_real, xq, baseline=nb, label='discrete', linewidth = 2)
    plot(fig,ax, midpoints, n_real, '*', markersize=7, label='inputs', color=colors.green,auto_scale=True)
    text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, f"A = {a}\nx$_0$ = {gam}$\mu$m", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

    k_max   = max(np.max(np.imag(nk)), np.max(np.imag(n_list)))
    ylabel = 'Im(n)'
    title = f''
    fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True, xlim=xlim,ylim=(-k_max/20,k_max*(1+1/20)))

    n_imag = np.imag(n_list)
    plot(fig,ax, xx, np.imag(nk), label='smooth', color=colors.red,auto_scale=True)
    ax.stairs(n_imag, xq, baseline=0, label='discrete', linewidth = 2)
    plot(fig,ax, midpoints, n_imag, '*', markersize=7, label='inputs', color=colors.green,auto_scale=True)
    ax.text(0.05, 0.95, f"A = {a}\nx$_0$ = {gam}$\mu$m", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

    xlabel = 'x ($\mu$m)'; ylabel = 'n'; y2label = 'k'
    title = f'Refractive Index'
    fig, ax, ax2 = plot_setup(xlabel,ylabel,True,y2label,title=title,yclr=colors.blue, y2clr=colors.red, figsize=(5,4),auto_scale=True, xlim=xlim)
    #ax.set_ylabel("n", rotation=0)
    #ax2.set_ylabel("k", rotation=0)
    plot(fig,ax, xx, np.real(nk), color=colors.blue,auto_scale=True)
    plot(fig,ax2, xx, np.imag(nk), color=colors.red,auto_scale=True)

    xlabel = 'x ($\mu$m)'; ylabel = '$\epsilon^{\prime}$'; y2label = '$\epsilon^{\prime \prime}$'
    title = f'Dielectric Function'
    fig, ax, ax2 = plot_setup(xlabel,ylabel,True,y2label,title=title,yclr=colors.blue, y2clr=colors.red, figsize=(5,4),auto_scale=True, xlim=xlim)
    #ax.set_ylabel('$\epsilon^{\prime}$', rotation=0)
    #ax2.set_ylabel('$\epsilon^{\prime \prime}$', rotation=0)
    plot(fig,ax, xx, np.real(ee), label='re(n)', color=colors.blue,auto_scale=True)
    plot(fig,ax2, xx, np.imag(ee), label='im(n)', color=colors.red,auto_scale=True)

# plotting function for refractive index
def eps_plot(xx, ee, xq, e_list, gam, a, nb, zoomed):

    if (zoomed):
        xlim = (xx[0]/10,xx[-1]/10)
    else:
        xlim = (xx[0],xx[-1])

    xlabel = 'x ($\mu$m)'; ylabel = 'Re($\epsilon$)'
    title = f''
    fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True, xlim=xlim)

    e_real = np.real(e_list)
    midpoints = (xq[:-1] + xq[1:]) / 2
    plot(fig,ax, xx, np.real(ee), label='smooth', color=colors.red,auto_scale=True)
    ax.stairs(e_real, xq, baseline=nb**2, label='discrete', linewidth = 2)
    plot(fig,ax, midpoints, e_real, '*', markersize=7, label='inputs', color=colors.green,auto_scale=True)

    eps_max   = max(np.max(np.imag(ee)), np.max(np.imag(e_list)))
    ylabel = 'Im($\epsilon$)'
    title = f''
    fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True, xlim=xlim,ylim=(-eps_max/20,eps_max*(1+1/20)))

    e_imag = np.imag(e_list)
    plot(fig,ax, xx, np.imag(ee), label='smooth', color=colors.red,auto_scale=True)
    ax.stairs(e_imag, xq, baseline=0, label='discrete', linewidth = 2)
    plot(fig,ax, midpoints, e_imag, '*', markersize=7, label='inputs', color=colors.green,auto_scale=True)

def smooth_gate_by_epsprime(eps_re, eps0, sigma):
    """
    Gate ~1 when eps_re > eps0, ~0 when eps_re < eps0.
    eps0: threshold (e.g., eps0 = 1.3**2)
    sigma: softness of threshold (smaller = harder slam)
    """
    return 0.5 * (1.0 + np.tanh((eps_re - eps0) / sigma))
  
def HT_help(k=8, nb=1.7, sx=1, delta=0.05, alpha=None, sigma=None, n0=1.3, plot_flag=True, zoomed=False):

    dx      = 1/(100*k)               # Step size in 'continuous' Lorentzian
    xmin    = -20/k
    xmax    = - xmin

    nx      = 1 + int(np.floor((xmax - xmin) / dx))
    xx      = np.linspace(xmin, xmax, nx)
    e_re    = logistic(xx,k,nb,sx)                    # Smooth Lorentzian curve

    # x: your coordinate array, n_re: your real index profile (logistic), same length
    e_re = e_re.astype(float)
    N = len(e_re)

    # 1) pick asymptotic constants from the ends (robustly)
    eL = np.mean(e_re[:max(4, N//100)])      # left asymptote ~ ZnS
    eR = np.mean(e_re[-max(4, N//100):])     # right asymptote ~ air

    # 2) pad with constants (make it much longer than the transition)
    pad = 4*N   # start with 2–6x N; bigger if needed
    e_pad = np.r_[np.full(pad, eL), e_re, np.full(pad, eR)]

    # 3) apply a gentle taper only at the *outer* ends so the periodic join is smooth
    #    (keep the interior essentially untouched)
    w = np.ones_like(e_pad)
    M = len(e_pad)
    taper_frac = 0.05  # 5% at each end; tune 0.02–0.1
    taper = tukey(M, alpha=2*taper_frac)    # Tukey is 1 in middle, cosine at ends
    # tukey() tapers BOTH ends; we want the same: suppress only outer ends
    w *= taper

    # taper around constant baselines so you don’t distort the interior step:
    # (this makes ends smoothly go to their constants in a way compatible with FFT wrap)
    e_pad2 = eL + (e_pad - eL)*w  # uses nL baseline; alternatively do piecewise baseline, see note below

    # 4) Hilbert transform and crop back to original region
    z = hilbert(e_pad2)
    e_im = np.imag(z)[pad:pad+N] 

    if alpha is not None:
        e_im = alpha*e_im
    if sigma is not None:
        e_im = e_im*smooth_gate_by_epsprime(e_re, n0**2, sigma)

    ee = e_re + 1j*e_im

    nk = np.sqrt(ee)

    e_scale = np.max(abs(ee-nb**2))
    x_scale = xx[-1] - xx[0]

    #discretizing algorithm for dielectric function
    count = 0
    xq, eq = [xx[0]], [ee[0]]
    for k in range(1, len(xx)):
        dx = (xx[k] - xq[-1]) / x_scale
        de = abs(ee[k] - eq[-1]) / e_scale
        ds = np.sqrt(dx**2 + de**2)
        
        if ds > delta:
            xq.append(xx[k])
            eq.append(ee[k])
            count = count + 1

    xq = np.append(xq, xmax)
    eq = np.append(eq, ee[-1])
    d_list = np.diff(xq)
    e_list = (eq[:-1] + eq[1:]) / 2
    n_list = np.sqrt(e_list)
    # plot imaginary and real part of refractive index
    if (plot_flag):
        nk_plot(xx, ee, nk, xq, n_list, 0.01, 10, nb, zoomed)
        eps_plot(xx, ee, xq, e_list, 0.01, 10, nb, zoomed)
    
    return (n_list.tolist(), d_list.tolist()) 


def generate_n_and_d_v6_symmetry(gam, a, nb, delta=0.02, domain_factor=200, plot_flag=False, zoomed=True):

    dx      = gam/100             # Step size in 'continuous' Lorentzian
    xmax    = gam * domain_factor  # Limits of Lorentzian

    nx      = 1 + int(np.floor(xmax / dx))
    xx      = np.linspace(0.0, xmax, nx)
    ee      = eps(xx,a,gam,nb)                    # Smooth Lorentzian curve
    nk      = np.sqrt(ee)

    e_scale   = np.max(abs(ee-nb**2))
    x_scale = xmax

    xq, eq = [xx[0]], [ee[0]]
    for k in range(1, len(xx)):
        dx = (xx[k] - xq[-1]) / x_scale
        de = np.abs(ee[k] - eq[-1]) / e_scale
        ds = np.sqrt(dx**2 + de**2)
        if ds > delta:
            xq.append(xx[k])
            eq.append(ee[k])

    xq = np.append(xq, xmax)
    eq = np.append(eq, ee[-1])

    xq_sym = np.concatenate([-xq[:0:-1], xq])  # mirror to full symmetric profile

    d_list = np.diff(xq_sym)
    log_term = np.log(xq_sym[1:] + 1j*gam) - np.log(xq_sym[:-1] + 1j*gam)
    e_list = nb**2 - a*gam * log_term / d_list
    n_list = np.sqrt(e_list)

    # plot imaginary and real part of refractive index
    if (plot_flag):
        xx_sym = np.concatenate([-xx[:0:-1], xx])
        ee_sym     = eps(xx_sym,a,gam,nb)
        nk_sym      = np.sqrt(ee_sym)
        nk_plot(xx_sym, ee_sym, nk_sym, xq_sym, n_list, gam, a, nb, zoomed)
        eps_plot(xx_sym, ee_sym, xq_sym, e_list, gam, a, nb, zoomed)
    
    return (n_list.tolist(), d_list.tolist())

def _make_c_list(n_list, d_list, lamb, angle=0, threshold=5):
    """Auto-generate coherent/incoherent classification for each layer.

    A layer is incoherent when its optical path n*d*cos(theta)/lambda exceeds
    the threshold — meaning Fabry-Perot fringes are too dense to resolve.

    Parameters
    ----------
    n_list : list of complex — refractive indices
    d_list : list of float — layer thicknesses (um)
    lamb : float — vacuum wavelength (um)
    angle : float — angle of incidence in radians (in medium 0)
    threshold : float — layers with n*d*cos(theta)/lambda > threshold are 'i'

    Returns
    -------
    c_list : list of str — 'c' (coherent) or 'i' (incoherent) per layer
    """
    n0_real = np.real(n_list[0])
    sin_angle = np.sin(angle)
    c_list = []
    for i, (n, d) in enumerate(zip(n_list, d_list)):
        if i == 0 or i == len(n_list) - 1 or d == inf:
            c_list.append('i')
        else:
            n_real = np.real(n)
            # Snell's law: cos(theta) inside this layer
            sin_ratio = n0_real * sin_angle / n_real
            cos_theta = np.sqrt(max(0, 1 - sin_ratio**2))
            optical_thickness = n_real * d * cos_theta / lamb
            c_list.append('i' if optical_thickness > threshold else 'c')
    return c_list


def TRA(n_list, d_list, lamb=3, angle=0, pol='p', threshold=5):

    c_list = _make_c_list(n_list, d_list, lamb, angle, threshold)
    result = tmm.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)
    T, R = result['T'], result['R']

    return (T, R, 1 - T - R)


def TRA_wavelength(n_list, d_list, lambda_list, angle=0, pol='p', threshold=5):

    T_list = np.zeros_like(lambda_list)
    R_list = np.zeros_like(lambda_list)
    A_list = np.zeros_like(lambda_list)

    for j, lamb in enumerate(lambda_list):
        c_list = _make_c_list(n_list, d_list, lamb, angle, threshold)
        result = tmm.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)
        T_list[j] = result['T']
        R_list[j] = result['R']
        A_list[j] = 1 - T_list[j] - R_list[j]

    return (T_list, R_list, A_list)


def TRA_angle(n_list, d_list, angle_list, lamb=3, pol='p', threshold=5):

    T_list = np.zeros_like(angle_list)
    R_list = np.zeros_like(angle_list)
    A_list = np.zeros_like(angle_list)

    for j, angle in enumerate(angle_list):
        c_list = _make_c_list(n_list, d_list, lamb, angle, threshold)
        result = tmm.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)
        T_list[j] = result['T']
        R_list[j] = result['R']
        A_list[j] = 1 - T_list[j] - R_list[j]

    return (T_list, R_list, A_list)


# Deprecated shims — c_list is now auto-generated internally
def TRA_inc(n_list, d_list, c_list=None, lamb=3, angle=0, pol='p', threshold=5):
    return TRA(n_list, d_list, lamb=lamb, angle=angle, pol=pol, threshold=threshold)

def TRA_wavelength_inc(n_list, d_list, c_list=None, lambda_list=None, angle=0, pol='p', threshold=5):
    return TRA_wavelength(n_list, d_list, lambda_list, angle=angle, pol=pol, threshold=threshold)

def TRA_angle_inc(n_list, d_list, c_list=None, angle_list=None, lamb=3, pol='p', threshold=5):
    return TRA_angle(n_list, d_list, angle_list, lamb=lamb, pol=pol, threshold=threshold)

def TRA_more(n_list, d_list, lambda_list, pol='p', angle=0):

    T_list = np.zeros_like(lambda_list)
    R_list = np.zeros_like(lambda_list)
    A_list = np.zeros_like(lambda_list)
    vw_list = np.empty((len(lambda_list), len(n_list), 2), dtype=complex)
    kz_list = np.empty((len(lambda_list), len(n_list)), dtype=complex)
    theta_list = np.empty((len(lambda_list), len(n_list)), dtype=complex)
    
    for j, lamb in enumerate(lambda_list):
        result = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)
        T_list[j] = result['T']
        R_list[j] = result['R']
        A_list[j] = 1 - T_list[j] - R_list[j]
        vw_list[j] = result['vw_list']
        kz_list[j] = result['kz_list']
        theta_list[j] = result['th_list']

    return (T_list, R_list, A_list, vw_list, kz_list, theta_list)