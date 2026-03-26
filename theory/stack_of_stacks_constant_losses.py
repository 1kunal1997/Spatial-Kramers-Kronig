# %%

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors
from matplotlib.ticker import MaxNLocator
import os
from matplotlib import colormaps

degrees = np.pi/180
# %% ##############################################################################################

#A = 10.9
#gam = 0.07
#nb = 2.3
A = 50
gam = 0.01
nb = 2.3

n_list, d_list = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, delta=0.1, plot_flag=False, zoomed=True)

# %% #############################################################################################

lamb = 5
angle = 0
losses_KK = np.sum(d_list * np.imag(n_list))
trans_bulk = np.exp(-4*np.pi*losses_KK/lamb)
emiss_bulk = 1 - trans_bulk
print(losses_KK)

# add semi-infinite air layers
d_list.append(np.inf)
d_list.insert(0, np.inf)
n_list.append(nb)       
n_list.insert(0, nb)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

T_LR_KK, R_LR_KK, A_LR_KK = tmm_h.TRA(n_list, d_list, lamb, angle=angle*degrees)
T_RL_KK, R_RL_KK, A_RL_KK = tmm_h.TRA(n_list_reversed, d_list_reversed, lamb, angle=angle*degrees)

# %%

#k_bulk = 1.13e-4
k_bulk = 1.04e-1
losses_bulk = 5000 * k_bulk
print(losses_bulk)
#losses_bulk = 0.52
N = int(losses_bulk/losses_KK) + 1
#N = 2
print(f'N: {N}')
I = np.empty(N, dtype=object)
C = (1/T_LR_KK) * np.array([[1, -R_RL_KK], [R_LR_KK, T_LR_KK**2 - R_LR_KK*R_RL_KK]])
for j in range(N):
    #print(j)
    losses_curr = (losses_bulk - j*losses_KK) / (j+1)
    print(f'Length of bulk: {losses_curr/k_bulk} um')
    P = np.array([[np.exp(4*np.pi*losses_curr/lamb), 0], [0, np.exp(-4*np.pi*losses_curr/lamb)]])

    stack = C @ P
    res = P
    for _ in range(j):
        res = res @ stack

    I[j] = res

# %% ####################################################################################

T, R_LR, R_RL, A_LR, A_RL, emiss_bulk, trans_bulk = (np.empty(N, dtype=float) for _ in range(7))
FOM_RL, FOM_LR, ASYM, FOM_bulk, FOM_enh = (np.empty(N, dtype=float) for _ in range(5))

for j in range(N):
    I_i = I[j]
    R_LR[j] = (I_i[1,0]/I_i[0,0])
    T[j] = (1/I_i[0,0])
    R_RL[j] = (-I_i[0,1]/I_i[0,0])
    A_LR[j] = (1 - T[j] - R_LR[j])
    A_RL[j] = (1 - T[j] - R_RL[j])

# %% ####################################################################################

np.savetxt(f'Data/sweep_numstacks_gam/A_RL_vs_numstacks_losses={losses_bulk}_A={A}_gam={gam}_nb={nb}_lam={lamb}_angle={angle}.txt', A_RL)

np.savetxt(f'Data/sweep_numstacks_gam/T_vs_numstacks_losses={losses_bulk}_A={A}_gam={gam}_nb={nb}_lam={lamb}_angle={angle}.txt', T)
# %% ####################################################################################
rnge = (A_RL[0] - A_RL[-1])/2
xlabel = 'Number of Stacks'; ylabel = ''
title = f'A = {A}, x$_0$ = {gam}'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(0,N-1),ylim=(T[int(N/2)]-rnge, T[int(N/2)]+rnge),figsize=(5,4),auto_scale=True)

plot(fig,ax,range(N),T, color=colors.blue, label='T',auto_scale=True)

#legend(fig,ax,auto_scale=True)

fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(0,N-1),figsize=(5,4),auto_scale=True)
plot(fig,ax,range(N),A_RL, color=colors.red, label='A_RL',auto_scale=True)
plot(fig,ax,range(N),A_LR, color=colors.light_red, label='A_LR',auto_scale=True)

# %% ####################################################################################

xlabel = 'Number of Stacks'; ylabel = 'Transmission'; y2label = 'Emission'
title = f'A = {A}, x$_0$ = {gam}$\mu m$'
ylim = (T[int(N/2)]-rnge, T[int(N/2)]+rnge)
fig, ax, ax2 = plot_setup(xlabel,ylabel,True,y2label,title=title,yclr=colors.blue, y2clr=colors.red, figsize=(5,4),auto_scale=True, xlim=(0,N-1), ylim=ylim)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plot(fig,ax,range(N),T,'-o', markersize=8, markevery=max(int(N/6),1), color=colors.blue, label='T',auto_scale=True)
plot(fig,ax2,range(N),A_RL,'-o', markersize=8, markevery=max(int(N/6),1), color=colors.red, label='A_RL',auto_scale=True)
#plot(fig,ax2,range(N),A_LR,'-o', markersize=8, markevery=max(int(N/6),1), color=colors.orange, label='A_RL',auto_scale=True)

# %%
print(A_RL)

# %%

def plot_param_sweep(
    y_vals, params, 
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
                         xlim=(0, len(y_vals[0])-1),
                         figsize=figsize, auto_scale=True)

    # Plot curves
    for i, p in enumerate(params):
        N = len(y_vals[i])
        color = cmap(i / (n_curves - 1)) if n_curves > 1 else cmap(0.5)
        plot(fig, ax, range(0,N), y_vals[i],'-o', markersize=6, markevery=max(int(N/6),1), color=color, label=f"{paramlabel}={p}", auto_scale=True)
        
    # Add legend
    legend(fig, ax, auto_scale=True)
# %% ##############################################################################################
data_dir = "Data/sweep_numstacks"

# Get all .txt files in the directory, sorted alphabetically
files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
sorted_files_by_size = sorted(files, key=lambda f: os.path.getsize(os.path.join(data_dir, f)), reverse=True)

print(len(files))
# Load all files into y_vals
y_vals = [np.loadtxt(os.path.join(data_dir, f)) for f in sorted_files_by_size]

for i, arr in enumerate(y_vals):
    print(np.min(arr))
# %% ##############################################################################################
params = [0.010, 0.012, 0.014, 0.018, 0.023, 0.035, 0.070]

plot_param_sweep(
    y_vals, params,
    xlabel="number of sKK stacks",
    ylabel="emission asymmetry",
    title=f"x$_0$={0.01}$\mu$m, n$_b$={nb}, losses={losses_bulk}$\mu$m",
    paramlabel="x$_0$",
    cmap_name="Reds")

tmm_h.show_textbox(text=f'n$_b$={nb}\nA={A}\nx$_0$={gam}', fontsize=14)

# %%
