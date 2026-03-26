import tmm

from numpy import pi, inf 
import matplotlib.pyplot as plt
import numpy as np

from plot_functions import plot_setup, plot, legend
import colors
from matplotlib import colormaps

# generate nk for spatial KK stack
def eps(x, a, gam, nb):
    return nb**2 - a * gam / (x + 1j*gam)

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
    ax.patch.set_alpha(0)

    # Display inline in VS Code interactive window
    plt.show()

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
    fig, ax, ax2 = plot_setup(xlabel,ylabel,True,y2label,title=title,yclr=colors.blue, y2clr=colors.red, figsize=(5,4),auto_scale=True, xlim=(xx[0]/10,xx[-1]/10))
    #ax.set_ylabel("n", rotation=0)
    #ax2.set_ylabel("k", rotation=0)
    plot(fig,ax, xx, np.real(nk), color=colors.blue,auto_scale=True)
    plot(fig,ax2, xx, np.imag(nk), color=colors.red,auto_scale=True)

    xlabel = 'x ($\mu$m)'; ylabel = '$\epsilon^{\prime}$'; y2label = '$\epsilon^{\prime \prime}$'
    title = f'Dielectric Function'
    fig, ax, ax2 = plot_setup(xlabel,ylabel,True,y2label,title=title,yclr=colors.blue, y2clr=colors.red, figsize=(5,4),auto_scale=True, xlim=(xx[0]/10,xx[-1]/10))
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
    text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, f"A = {a}\nx$_0$ = {gam}$\mu$m", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

    eps_max   = max(np.max(np.imag(ee)), np.max(np.imag(e_list)))
    ylabel = 'Im($\epsilon$)'
    title = f''
    fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True, xlim=xlim,ylim=(-eps_max/20,eps_max*(1+1/20)))

    e_imag = np.imag(e_list)
    plot(fig,ax, xx, np.imag(ee), label='smooth', color=colors.red,auto_scale=True)
    ax.stairs(e_imag, xq, baseline=0, label='discrete', linewidth = 2)
    plot(fig,ax, midpoints, e_imag, '*', markersize=7, label='inputs', color=colors.green,auto_scale=True)
    ax.text(0.05, 0.95, f"A = {a}\nx$_0$ = {gam}$\mu$m", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)
    

# function to generate list of refractive indices and thicknesses of each layer in TMM calculation
def generate_n_and_d(gam, a, nb, del_n_prop = 1/25, del_x_prop = 15, min_thickness=0.001, plot_flag=False, zoomed=True):
    
    dx      = gam/100               # Step size in 'continuous' Lorentzian
    xmin    = -gam * 200           # Limits of Lorentzian
    xmax    = - xmin

    nx      = 1 + int(np.floor((xmax - xmin) / dx))
    xx      = np.linspace(xmin, xmax, nx)
    ee      = eps(xx,a,gam,nb)                    # Smooth Lorentzian curve
    nk      = np.sqrt(ee)

    k_max   = np.max(np.imag(nk))     # Max k value. used to set max n-step size 

    del_n   = k_max*del_n_prop     # k_max/25. Max n-step size in discrete Lorentzian approximation
    del_x   = gam*del_x_prop     # 15*gam. Max x-step size in discrete Lorentzian approximation

    xq      = [xx[0]]                               
    nq      = [nk[0]]
    count   = 0

    for k in range(0,nx):
        if (abs((nk[k]) - (nq[count])) > del_n and abs((xx[k]) - (xq[count])) > min_thickness or 
            abs((xx[k]) - (xq[count])) > del_x):
            xq.append(xx[k])
            nq.append(nk[k])
            count = count + 1

    xq = np.append(xq,xmax)     # should we be appending xx[-1]? because xx does not include xmax as it is rn
    nq = np.append(nq,nk[-1])

    d_list = np.diff(xq)
    n_list = (nq[:-1] + nq[1:]) / 2
    
    # plot imaginary and real part of refractive index
    if (plot_flag):
        nk_plot(xx, ee, nk, xq, n_list, gam, a, nb, zoomed)
    
    return (n_list.tolist(), d_list.tolist())

# function to generate list of refractive indices and thicknesses of each layer in TMM calculation
def generate_n_and_d_new(gam, a, nb, delta=0.02, plot_flag=False, zoomed=True):
    
    dx      = gam/100               # Step size in 'continuous' Lorentzian
    xmin    = -gam * 200           # Limits of Lorentzian
    xmax    = - xmin

    nx      = 1 + int(np.floor((xmax - xmin) / dx))
    xx      = np.linspace(xmin, xmax, nx)
    ee      = eps(xx,a,gam,nb)                    # Smooth Lorentzian curve
    nk      = np.sqrt(ee)

    n_scale   = np.max(abs(nk-nb))
    x_scale = xmax - xmin
    #x_scale = gam*400
    
    '''
    #if you want to weigh curvature into interpolation schema 
    dn_dx = np.gradient(nk, xx)
    d2n_dx2 = np.gradient(dn_dx, xx)
    curvature_scale = np.max(abs(d2n_dx2))
    '''
    beta = 0
    curvature_boost = 0

    count = 0
    xq, nq = [xx[0]], [nk[0]]
    for k in range(1, len(xx)):
        dx = (xx[k] - xq[-1]) / x_scale
        dn = abs(nk[k] - nq[-1]) / n_scale
        ds = np.sqrt(dx**2 + dn**2)
        
        #curvature_boost = abs(d2n_dx2[k]) / curvature_scale
        if ds * (1 + beta * curvature_boost) > delta:
            xq.append(xx[k])
            nq.append(nk[k])
            count = count + 1

    #print(f'Number of Layers: {count}')
    xq = np.append(xq,xmax)     # should we be appending xx[-1]? because xx does not include xmax as it is rn
    nq = np.append(nq,nk[-1])

    #print(f'xq is: {xq}')
    #print(f'nq is: {nq}')

    d_list = np.diff(xq)
    n_list = (nq[:-1] + nq[1:]) / 2

    #print(f'd_list is: {d_list}')
    #print(f'n_list is: {n_list}')
    
    # plot imaginary and real part of refractive index
    if (plot_flag):
        nk_plot(xx, ee, nk, xq, n_list, gam, a, nb, zoomed)
    
    return (n_list.tolist(), d_list.tolist())

# made change to discretize eps instead of n to make symmetric discretization
def generate_n_and_d_v3(gam, a, nb, delta=0.02, plot_flag=False, zoomed=True):
    
    dx      = gam/100               # Step size in 'continuous' Lorentzian
    xmin    = -gam * 200           # Limits of Lorentzian
    xmax    = - xmin

    nx      = 1 + int(np.floor((xmax - xmin) / dx))
    xx      = np.linspace(xmin, xmax, nx)
    ee      = eps(xx,a,gam,nb)                    # Smooth Lorentzian curve
    nk      = np.sqrt(ee)

    e_scale   = np.max(abs(ee-nb**2))
    x_scale = xmax - xmin
    #x_scale = gam*400
    
    '''
    #if you want to weigh curvature into interpolation schema 
    dn_dx = np.gradient(nk, xx)
    d2n_dx2 = np.gradient(dn_dx, xx)
    curvature_scale = np.max(abs(d2n_dx2))
    '''
    beta = 0
    curvature_boost = 0

    count = 0
    xq, eq = [xx[0]], [ee[0]]
    for k in range(1, len(xx)):
        dx = (xx[k] - xq[-1]) / x_scale
        de = abs(ee[k] - eq[-1]) / e_scale
        ds = np.sqrt(dx**2 + de**2)
        
        #curvature_boost = abs(d2n_dx2[k]) / curvature_scale
        if ds * (1 + beta * curvature_boost) > delta:
            xq.append(xx[k])
            eq.append(ee[k])
            count = count + 1

    #print(f'Number of Layers: {count}')
    xq = np.append(xq,xmax)     # should we be appending xx[-1]? because xx does not include xmax as it is rn
    eq = np.append(eq,ee[-1])
    #print(f'xq is: {xq}')
    #print(f'eq is: {eq}')
    d_list = np.diff(xq)
    e_list = (eq[:-1] + eq[1:]) / 2
    #print(f'd_list is: {d_list}')
    #print(f'e_list is: {e_list}')
    n_list = np.sqrt(e_list)
    #print(f'n_list is: {n_list}')

    # plot imaginary and real part of refractive index
    if (plot_flag):
        nk_plot(xx, ee, nk, xq, n_list, gam, a, nb, zoomed)
        eps_plot(xx, ee, xq, e_list, gam, a, nb, zoomed)
    
    return (n_list.tolist(), d_list.tolist())

def generate_n_and_d_v4_xmid(gam, a, nb, delta=0.02, plot_flag=False, zoomed=True):
    
    dx      = gam/100               # Step size in 'continuous' Lorentzian
    xmin    = -gam * 200           # Limits of Lorentzian
    xmax    = - xmin

    nx      = 1 + int(np.floor((xmax - xmin) / dx))
    xx      = np.linspace(xmin, xmax, nx)
    ee      = eps(xx,a,gam,nb)                    # Smooth Lorentzian curve
    nk      = np.sqrt(ee)

    e_scale   = np.max(abs(ee-nb**2))
    x_scale = xmax - xmin
    #x_scale = gam*400
    
    '''
    #if you want to weigh curvature into interpolation schema 
    dn_dx = np.gradient(nk, xx)
    d2n_dx2 = np.gradient(dn_dx, xx)
    curvature_scale = np.max(abs(d2n_dx2))
    '''
    beta = 0
    curvature_boost = 0

    count = 0
    xq, eq = [xx[0]], [ee[0]]
    for k in range(1, len(xx)):
        dx = (xx[k] - xq[-1]) / x_scale
        de = abs(ee[k] - eq[-1]) / e_scale
        ds = np.sqrt(dx**2 + de**2)
        
        #curvature_boost = abs(d2n_dx2[k]) / curvature_scale
        if ds * (1 + beta * curvature_boost) > delta:
            xq.append(xx[k])
            eq.append(ee[k])
            count = count + 1

    #print(f'Number of Layers: {count}')
    xq = np.append(xq,xmax)     # should we be appending xx[-1]? because xx does not include xmax as it is rn
    eq = np.append(eq,ee[-1])
    #print(f'xq is: {xq}')
    #print(f'eq is: {eq}')
    x_list = (xq[:-1] + xq[1:]) / 2
    d_list = np.diff(xq)
    #e_list = (eq[:-1] + eq[1:]) / 2
    e_list = eps(x_list, a, gam, nb)
    #print(f'd_list is: {d_list}')
    #print(f'e_list is: {e_list}')
    n_list = np.sqrt(e_list)
    #print(f'n_list is: {n_list}')

    # plot imaginary and real part of refractive index
    if (plot_flag):
        nk_plot(xx, ee, nk, xq, n_list, gam, a, nb, zoomed)
        eps_plot(xx, ee, xq, e_list, gam, a, nb, zoomed)
    
    return (n_list.tolist(), d_list.tolist())

def generate_n_and_d_v5_avg_over_cell(gam, a, nb, delta=0.02, plot_flag=False, zoomed=True):
    
    dx      = gam/100               # Step size in 'continuous' Lorentzian
    xmin    = -gam * 200           # Limits of Lorentzian
    xmax    = - xmin

    nx      = 1 + int(np.floor((xmax - xmin) / dx))
    xx      = np.linspace(xmin, xmax, nx)
    ee      = eps(xx,a,gam,nb)                    # Smooth Lorentzian curve
    nk      = np.sqrt(ee)

    e_scale   = np.max(abs(ee-nb**2))
    x_scale = xmax - xmin
    #x_scale = gam*400
    
    '''
    #if you want to weigh curvature into interpolation schema 
    dn_dx = np.gradient(nk, xx)
    d2n_dx2 = np.gradient(dn_dx, xx)
    curvature_scale = np.max(abs(d2n_dx2))
    '''
    beta = 0
    curvature_boost = 0

    count = 0
    xq, eq = [xx[0]], [ee[0]]
    for k in range(1, len(xx)):
        dx = (xx[k] - xq[-1]) / x_scale
        de = abs(ee[k] - eq[-1]) / e_scale
        ds = np.sqrt(dx**2 + de**2)
        
        #curvature_boost = abs(d2n_dx2[k]) / curvature_scale
        if ds * (1 + beta * curvature_boost) > delta:
            xq.append(xx[k])
            eq.append(ee[k])
            count = count + 1

    #print(f'Number of Layers: {count}')
    xq = np.append(xq,xmax)     # should we be appending xx[-1]? because xx does not include xmax as it is rn
    eq = np.append(eq,ee[-1])
    x_list = (xq[:-1] + xq[1:]) / 2
    d_list = np.diff(xq)

    log_term = np.log(xq[1:] + 1j*gam) - np.log(xq[:-1] + 1j*gam)
    e_list = nb**2 - a*gam * log_term / d_list
    n_list = np.sqrt(e_list)
    
    for i, x in enumerate(x_list):
        print(f'x: {x}, imag(eps): {e_list[i].imag}')
    print('\n')
    for i, x in enumerate(x_list):
        print(f'x: {x}, imag(n): {n_list[i].imag}')
    
    # plot imaginary and real part of refractive index
    if (plot_flag):
        nk_plot(xx, ee, nk, xq, n_list, gam, a, nb, zoomed)
        eps_plot(xx, ee, xq, e_list, gam, a, nb, zoomed)
    
    return (n_list.tolist(), d_list.tolist())

def generate_n_and_d_v6_symmetry(gam, a, nb, delta=0.02, plot_flag=False, zoomed=True):
    
    dx      = gam/1000               # Step size in 'continuous' Lorentzian
    xmax    = gam * 2000           # Limits of Lorentzian

    nx      = 1 + int(np.floor(xmax / dx))
    xx      = np.linspace(0.0, xmax, nx)
    ee      = eps(xx,a,gam,nb)                    # Smooth Lorentzian curve
    nk      = np.sqrt(ee)

    e_scale   = np.max(abs(ee-nb**2))
    x_scale = xmax
    #x_scale = gam*400
    
    '''
    #if you want to weigh curvature into interpolation schema 
    dn_dx = np.gradient(nk, xx)
    d2n_dx2 = np.gradient(dn_dx, xx)
    curvature_scale = np.max(abs(d2n_dx2))
    '''
    beta = 0
    curvature_boost = 0

    count = 1
    xq, eq = [xx[0]], [ee[0]]
    for k in range(1, len(xx)):
        dx = (xx[k] - xq[-1]) / x_scale
        de = np.abs(ee[k] - eq[-1]) / e_scale
        ds = np.sqrt(dx**2 + de**2)
        
        #curvature_boost = abs(d2n_dx2[k]) / curvature_scale
        if ds * (1 + beta * curvature_boost) > delta:
            xq.append(xx[k])
            eq.append(ee[k])
            count = count + 1

    xq = np.append(xq,xmax)     # should we be appending xx[-1]? because xx does not include xmax as it is rn
    eq = np.append(eq,ee[-1])
    count = count + 1
    print(f'Number of Layers: {count}\n')

    '''
    for i, x in enumerate(xq):
        print(f'xq: {x}, re(eps): {eq[i].real}, imag(eps): {eq[i].imag}')
    print('\n')
    '''

    xq_sym = np.concatenate([-xq[:0:-1], xq])

    '''
    for i, x in enumerate(xq_sym):
        print(f'xq_sym: {x}')
    print('\n')
    '''

    x_list = (xq_sym[:-1] + xq_sym[1:]) / 2
    d_list = np.diff(xq_sym)

    log_term = np.log(xq_sym[1:] + 1j*gam) - np.log(xq_sym[:-1] + 1j*gam)
    e_list = nb**2 - a*gam * log_term / d_list
    n_list = np.sqrt(e_list)

    '''
    for i, x in enumerate(x_list):
        print(f'x: {x}, imag(eps): {e_list[i].imag}')
    print('\n')
    for i, x in enumerate(x_list):
        print(f'x: {x}, imag(n): {n_list[i].imag}')
    '''

    # plot imaginary and real part of refractive index
    if (plot_flag):
        xx_sym = np.concatenate([-xx[:0:-1], xx])
        ee_sym     = eps(xx_sym,a,gam,nb)
        nk_sym      = np.sqrt(ee_sym)
        nk_plot(xx_sym, ee_sym, nk_sym, xq_sym, n_list, gam, a, nb, zoomed)
        eps_plot(xx_sym, ee_sym, xq_sym, e_list, gam, a, nb, zoomed)
    
    return (n_list.tolist(), d_list.tolist())

def TRA(n_list, d_list, lamb=3, angle=0, pol='p'):

    T = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['T']
    R = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['R']
    A = 1 - T - R

    return (T, R, A)

def TRA_inc(n_list, d_list, c_list, lamb=3, angle=0, pol='p'):

    T = tmm.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)['T']
    R = tmm.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)['R']
    A = 1 - T - R

    return (T, R, A)

def TRA_wavelength(n_list, d_list, lambda_list, angle=0, pol='p'):

    T_list = np.zeros_like(lambda_list)
    R_list = np.zeros_like(lambda_list)
    A_list = np.zeros_like(lambda_list)
    
    for j, lamb in enumerate(lambda_list):
        T_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['T']
        R_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['R']
        A_list[j] = 1 - T_list[j] - R_list[j]

    return (T_list, R_list, A_list)

def TRA_wavelength_inc(n_list, d_list, c_list, lambda_list, angle=0, pol='p'):

    T_list = np.zeros_like(lambda_list)
    R_list = np.zeros_like(lambda_list)
    A_list = np.zeros_like(lambda_list)
    
    for j, lamb in enumerate(lambda_list):
        T_list[j] = tmm.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)['T']
        R_list[j] = tmm.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)['R']
        A_list[j] = 1 - T_list[j] - R_list[j]

    return (T_list, R_list, A_list)

def TRA_angle(n_list, d_list, angle_list, lamb=3, pol='p'):

    T_list = np.zeros_like(angle_list)
    R_list = np.zeros_like(angle_list)
    A_list = np.zeros_like(angle_list)
    
    for j, angle in enumerate(angle_list):
        O = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)
        T_list[j] = O['T']
        R_list[j] = O['R']
        A_list[j] = 1 - T_list[j] - R_list[j]

    return (T_list, R_list, A_list)

def TRA_angle_inc(n_list, d_list, c_list, angle_list, lamb=3, pol='p'):

    T_list = np.zeros_like(angle_list)
    R_list = np.zeros_like(angle_list)
    A_list = np.zeros_like(angle_list)
    
    for j, angle in enumerate(angle_list):
        T_list[j] = tmm.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)['T']
        R_list[j] = tmm.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)['R']
        A_list[j] = 1 - T_list[j] - R_list[j]

    return (T_list, R_list, A_list)

def TRA_more(n_list, d_list, lambda_list, pol='p', angle=0):

    T_list = np.zeros_like(lambda_list)
    R_list = np.zeros_like(lambda_list)
    A_list = np.zeros_like(lambda_list)
    vw_list = np.empty((len(lambda_list), len(n_list), 2), dtype=complex)
    kz_list = np.empty((len(lambda_list), len(n_list)), dtype=complex)
    theta_list = np.empty((len(lambda_list), len(n_list)), dtype=complex)
    
    for j, lamb in enumerate(lambda_list):
        T_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['T']
        R_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['R']
        A_list[j] = 1 - T_list[j] - R_list[j]
        vw_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['vw_list']
        kz_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['kz_list']
        theta_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['th_list']

    return (T_list, R_list, A_list, vw_list, kz_list, theta_list)