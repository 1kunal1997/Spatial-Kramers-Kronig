# %%

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors
from matplotlib import colormaps
import matplotlib.colors as mcolors

#%% ###############################################################################################

def plot_param_sweep(
    width_arr, y_vals, params,
    xlabel="X", ylabel="Y", title="",
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

    # Always normalize linearly for consistent distribution
    norm = mcolors.Normalize(vmin=params.min(), vmax=params.max())

    # Set up figure
    fig, ax = plot_setup(xlabel, ylabel, title=title,
                         xlim=(width_arr[0], width_arr[-1]),
                         figsize=figsize, auto_scale=True)

    # Plot curves
    for i, p in enumerate(params):
        color = cmap(norm(p))
        plot(fig, ax, width_arr, y_vals[i], color=color, label=f"A={p}", auto_scale=True)

    # Add legend
    legend(fig, ax, auto_scale=True)

# %% ##############################################################################################

def generate_n_and_d_new(gam, a, nb, delta=0.02, width = 0, plot_flag=False, zoomed=True):

    dx      = gam/100               # Step size in 'continuous' Lorentzian
    xmin    = -gam * 200 - width/2          # Limits of Lorentzian
    xmax    = - xmin

    nx      = 1 + int(np.floor((xmax - xmin) / dx))
    xx      = np.linspace(xmin, xmax, nx)
    ee      = tmm_h.eps(xx,a,gam,nb)                    # Smooth Lorentzian curve
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

    idx = len(xq) // 2
    xq[:idx] -= width/2
    xq[idx:] += width/2

    d_list = np.diff(xq)
    n_list = (nq[:-1] + nq[1:]) / 2

    #n_list, d_list, xq = stretch_middle_layer_centered(n_list, d_list, width)

    #idx = len(d_list) // 2
    #d_list[idx] = width
    #xq -= width
    '''
    width = 0
    center = int(len(d_list)/2)
    k_max = max(n_list.imag)
    #print(center)
    #print(k_max)
    for i in range(-width, width):
        n_list.real[center + i] = nb
        n_list.imag[center + i] = k_max

    #print(n_array)
    #print(d_array)
    '''
    # plot imaginary and real part of refractive index
    if (plot_flag):
        tmm_h.nk_plot(xx, ee, nk, xq, n_list, gam, a, nb, zoomed)

    return (n_list.tolist(), d_list.tolist())

def generate_n_and_d_v6_symmetry(gam, a, nb, delta=0.02, width=0, plot_flag=False, zoomed=True):

    dx      = gam/100               # Step size in 'continuous' Lorentzian
    xmax    = gam * 200           # Limits of Lorentzian

    nx      = 1 + int(np.floor(xmax / dx))
    xx      = np.linspace(0.0, xmax, nx)
    ee      = tmm_h.eps(xx,a,gam,nb)                    # Smooth Lorentzian curve
    #nk      = np.sqrt(ee)

    e_scale   = np.max(abs(ee-nb**2))
    x_scale = xmax

    count = 1
    xq, eq = [xx[0]], [ee[0]]
    for k in range(1, len(xx)):
        dx = (xx[k] - xq[-1]) / x_scale
        de = np.abs(ee[k] - eq[-1]) / e_scale
        ds = np.sqrt(dx**2 + de**2)

        #curvature_boost = abs(d2n_dx2[k]) / curvature_scale
        if ds > delta:
            xq.append(xx[k])
            eq.append(ee[k])
            count = count + 1

    xq = np.append(xq,xmax)     # should we be appending xx[-1]? because xx does not include xmax as it is rn
    eq = np.append(eq,ee[-1])
    count = count + 1
    print(f'Number of Layers: {count}')

    xq_window = xq

    '''
    for i, x in enumerate(xq):
        print(f'xq: {x}, imag(eps): {eq[i].imag}')
    print('\n')
    '''

    xq_sym = np.concatenate([-xq[:0:-1], xq])

    '''
    for i, x in enumerate(xq_sym):
        print(f'xq_sym: {x}')
    print('\n')
    '''

    d_list = np.diff(xq_sym)

    log_term = np.log(xq_sym[1:] + 1j*gam) - np.log(xq_sym[:-1] + 1j*gam)
    e_list = nb**2 - a*gam * log_term / d_list
    #n_list = np.sqrt(e_list)

    # add window
    e_window = nb**2 + 1j*A
    print(len(d_list))
    d_list_window = np.insert(d_list, len(d_list)//2, width)
    e_list_window = np.insert(e_list, len(d_list)//2, e_window)
    n_list_window = np.sqrt(e_list_window)

    # plot imaginary and real part of refractive index
    if (plot_flag):
        xx_sym = np.concatenate([-xx[:0:-1], xx])
        ee_sym     = tmm_h.eps(xx_sym,a,gam,nb)
        nk_sym      = np.sqrt(ee_sym)
        xq_sym_window = -0.5*np.sum(d_list_window) + np.concatenate(([0.0], np.cumsum(d_list_window)))
        tmm_h.nk_plot(xx_sym, ee_sym, nk_sym, xq_sym_window, n_list_window, gam, a, nb, zoomed)
        tmm_h.eps_plot(xx_sym, ee_sym, xq_sym_window, e_list_window, gam, a, nb, zoomed)

    return (n_list_window.tolist(), d_list_window.tolist())
# %% ##############################################################################################

A = 3
gam = 0.01
nb = 1.7

n_list, d_list = generate_n_and_d_v6_symmetry(gam, A, nb, delta=0.5, width=0.1, plot_flag=True, zoomed=True)
print(len(d_list))
#n_list, d_list = generate_n_and_d_new(gam, A, nb, delta=0.1, width=0.1, plot_flag=True, zoomed=True)

# %% #############################################################################################

lamb = 3
width_arr = np.arange(0, 1, 0.01)
T_LR, T_RL, R_LR, R_RL, A_LR, A_RL, emiss_bulk, trans_bulk = (np.empty(len(width_arr), dtype=float) for i in range(8))
FOM_RL, FOM_LR, ASYM, FOM_bulk, FOM_enh = (np.empty(len(width_arr), dtype=float) for i in range(5))

for i, width in enumerate(width_arr):
    print(i)
    n_list, d_list = generate_n_and_d_new(gam, A, nb, delta=0.01, width=width, plot_flag=False)
    losses_total = np.sum(d_list * np.imag(n_list))
    trans_bulk[i] = np.exp(-4*np.pi*losses_total/lamb)
    emiss_bulk[i] = 1 - trans_bulk[i]

    # add semi-infinite air layers
    d_list.append(np.inf)
    d_list.insert(0, np.inf)
    n_list.append(nb)
    n_list.insert(0, nb)

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]

    T_LR[i], R_LR[i], A_LR[i] = tmm_h.TRA(n_list, d_list, lamb)
    T_RL[i], R_RL[i], A_RL[i] = tmm_h.TRA(n_list_reversed, d_list_reversed, lamb)

    FOM_LR[i] = T_LR[i]**2 / A_LR[i]
    FOM_RL[i] = T_RL[i]**2 / A_RL[i]
    ASYM[i] = A_LR[i] / A_RL[i]
    FOM_bulk[i] = trans_bulk[i]**2 / emiss_bulk[i]
    FOM_enh[i] = FOM_RL[i] / FOM_bulk[i]

np.savetxt('Data/asym_A=3_gam=0.01_nb=1.7_delta=0.01.txt', ASYM)

# %%

A_3 = np.loadtxt(f'Data/asym_A=3_gam=0.01_nb=1.7_delta=0.5.txt')
A_4 = np.loadtxt(f'Data/asym_A=3_gam=0.01_nb=1.7_delta=0.2.txt')
A_5 = np.loadtxt(f'Data/asym_A=3_gam=0.01_nb=1.7_delta=0.1.txt')
A_6 = np.loadtxt(f'Data/asym_A=3_gam=0.01_nb=1.7_delta=0.05.txt')
A_7 = np.loadtxt(f'Data/asym_A=3_gam=0.01_nb=1.7_delta=0.01.txt')

xlabel = 'Number of Stacks'; ylabel = 'FOM enhancement'
title = f'A=5, n$_b$=1.7'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(width_arr[0],width_arr[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,width_arr,A_3, color=colors.blue, label='x$_0$=0.7nm',auto_scale=True)
plot(fig,ax,width_arr,A_4, color=colors.green, label='x$_0$=0.9nm',auto_scale=True)
plot(fig,ax,width_arr,A_5, color=colors.red, label='x$_0$=1.0nm',auto_scale=True)
plot(fig,ax,width_arr,A_6, color=colors.purple, label='x$_0$=2.0nm',auto_scale=True)
plot(fig,ax,width_arr,A_7, color=colors.copper, label='x$_0$=2.2nm',auto_scale=True)
#plot(fig,ax,range(N),np.log(FOM_bulk), '--', color=colors.red, label='bulk',auto_scale=True)

legend(fig,ax,auto_scale=True)

# %%

# Choose a colormap (here 'Blues' for your blue gradient example)
base_cmap = colormaps['Blues']
cmap = lambda x: base_cmap(0.15 + 0.85 * x)

#params = [0.01, 0.05, 0.1, 0.2, 0.5]
params = [3, 4, 5, 6, 7]
y_vals = [A_3, A_4, A_5, A_6, A_7]
# Normalize the parameter values to [0,1] for the colormap
norm = mcolors.Normalize(vmin=params[0], vmax=params[-1])

xlabel = 'Number of Stacks'; ylabel = 'FOM enhancement'
title = f''
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(width_arr[0],width_arr[-1]),figsize=(5,4),auto_scale=True)

for i, p in enumerate(params):
    color = cmap(norm(p))  # Map param value to a color
    plot(fig,ax,width_arr,y_vals[i], color=color, label=f'A={p}',auto_scale=True)

legend(fig,ax,auto_scale=True)

# %%

# === Example Usage ===
# Linear sweep

plot_param_sweep(
    width_arr, y_vals, params,
    xlabel="width of window (um)",
    ylabel="emission asymmetry",
    cmap_name="Reds", cmap_range=(0, 1))

# %% ############################################################################################

# plotting using Will's plot modules

xlabel = 'width of window'; ylabel = 'ASYM'
title = f''
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(width_arr[0],width_arr[99]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,width_arr[:100],FOM_enh[:100],label=r'FOM enhancement',color=colors.blue,auto_scale=True)
plot(fig,ax,width_arr[:99],ASYM[:99],label=r'Asymmetry',color=colors.green,auto_scale=True)
#plot(fig,ax,width_arr[:100],FOM_RL[:100],label=r'KK',color=colors.blue,auto_scale=True)
#plot(fig,ax,width_arr[:100],FOM_bulk[:100],label=r'bulk',color=colors.green,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

#plot(fig,ax,lambda_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

#plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
#plot(fig,ax,lambda_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)
for i, width in enumerate(width_arr):
    print(f'Width: {width}, ASYM: {ASYM[i]}')

# %% ##############################################################################################

A = 3
gam = 0.01
nb = 1.7
lamb = 3
degrees = np.pi/180

n_list, d_list = generate_n_and_d_new(gam, A, nb, delta=0.2, width=0.2, plot_flag=True, zoomed=False)

#for i, n in enumerate(n_list):
#    print(f'n:{n.real}, k:{n.imag}, d:{d_list[i]}')
#lambda_list = np.linspace(2,5,100)
angle_list = np.arange(0, 80, 1)
losses_total = np.sum(d_list * np.imag(n_list))
trans_bulk = np.exp(-4*np.pi*losses_total/(lamb*np.cos(angle_list*degrees)))
emiss_bulk = 1 - trans_bulk

# add semi-infinite air layers
d_list.append(np.inf)
d_list.insert(0, np.inf)
n_list.append(nb)
n_list.insert(0, nb)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle(n_list, d_list, angle_list*degrees, lamb=lamb, pol='s')
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_angle(n_list_reversed, d_list_reversed, angle_list*degrees, lamb=lamb, pol='s')
# %% ############################################################################################

# plotting using Will's plot modules

xlabel = 'Angle (degrees)'; ylabel = 'Fraction of Power'
title = f'TRA (A={A}, x$_0$={gam}$\mu$m)'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(angle_list[0],angle_list[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,angle_list,A_list_LR / A_list_RL,label=r'asymmetry',color=colors.blue,auto_scale=True)
plot(fig,ax,angle_list,T_list_LR, label=r'T$_{LR}$',color=colors.blue,auto_scale=True)
plot(fig,ax,angle_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,angle_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,angle_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,angle_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)
plot(fig,ax,angle_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

#legend(fig,ax,auto_scale=True)

# %% ############################################################################################

xlabel = 'Angle (degrees)'; ylabel = 'Asymmetry'
title = f'Emission Asymmetry (A={A}, x$_0$={gam}$\mu$m)'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(angle_list[0],angle_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,angle_list,A_list_LR / A_list_RL,label=r'asymmetry',color=colors.blue,auto_scale=True)
# %%
for i, angle in enumerate(angle_list):
    print(A_list_LR / A_list_RL)
# %%
