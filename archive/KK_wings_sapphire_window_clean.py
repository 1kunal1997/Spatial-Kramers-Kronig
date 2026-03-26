# %%

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors
import os
import matplotlib.pyplot as plt

# %% ##############################################################################################
def generate_n_and_d(gam, a, nb, delta=0.02, width=0, plot_flag=False, zoomed=True):

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

        if ds > delta:
            xq.append(xx[k])
            eq.append(ee[k])
            count = count + 1

    xq = np.append(xq,xmax)
    eq = np.append(eq,ee[-1])
    count = count + 1
    print(f'Number of Layers: {count}')

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

    # add window
    #e_window = nb**2 + 1j*A
    e_window = (1.79 + 1j*1.29)**2
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

A = 2.5
gam = 0.2
nb = 1.13

n_list, d_list = generate_n_and_d(gam, A, nb, delta=0.1, width=2, plot_flag=True, zoomed=True)
print(len(d_list))

# %% #############################################################################################

A = 2.5
gam = 0.2
nb = 1.13
lamb = 13
width_arr = np.arange(0.01, 4, 0.01)
delta = 0.1

T_LR, T_RL, R_LR, R_RL, A_LR, A_RL, emiss_bulk, trans_bulk = (np.empty(len(width_arr), dtype=float) for i in range(8))
FOM_RL, FOM_LR, ASYM, FOM_bulk, FOM_enh = (np.empty(len(width_arr), dtype=float) for i in range(5))

for i, width in enumerate(width_arr):
    print(i)
    n_list, d_list = generate_n_and_d(gam, A, nb, delta=delta, width=width, plot_flag=False)
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

# %% ##############################################################################################

#np.savetxt(f'Data/asym_A=2.5_gam=0.2_nb=1.13_nw=1.79_kw=1.29_lam=13/delta={delta}.txt', ASYM)

# %% ############################################################################################

xlabel = 'width of window ($\mu$m)'; ylabel = 'Emission Asymmetry'; y2label = 'Transmission'
title = f''
fig, ax, ax2 = plot_setup(xlabel,ylabel,True,y2label,title=title,yclr=colors.red, y2clr=colors.blue, figsize=(5,4),auto_scale=True, xlim=(width_arr[0],width_arr[-1]))

plot(fig,ax,width_arr,ASYM,label=r'Asym.',color=colors.red,auto_scale=True)
plot(fig,ax2,width_arr,T_LR,label=r'T',color=colors.blue,auto_scale=True)

#legend(fig,ax,auto_scale=True)

for i, width in enumerate(width_arr):
    print(f'Width: {width}, ASYM: {ASYM[i]}')

# %% ##############################################################################################
data_dir = "Data/asym_A=2.5_gam=0.2_nb=1.13_nw=1.79_kw=1.29_lam=13"

# Get all .txt files in the directory, sorted alphabetically
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")])
print(files)
# Load all files into y_vals
y_vals = [np.loadtxt(os.path.join(data_dir, f)) for f in files]

for i, arr in enumerate(y_vals):
    print(np.max(arr))
# %% ##############################################################################################
params = [10, 20, 50, 96, 420]

tmm_h.plot_param_sweep(
    width_arr, y_vals[::-1], params,
    xlabel="width of window ($\mu$m)",
    ylabel="emission asymmetry",
    title="",
    paramlabel="N$_{\ell}$",
    cmap_name="Reds")

tmm_h.show_textbox(text=f'n$_b$={nb}\nA={A}\nx$_0$={gam}', fontsize=14)

# %% ##############################################################################################

A = 3
gam = 0.01
nb = 1.7
lamb = 3
degrees = np.pi/180

n_list, d_list = generate_n_and_d(gam, A, nb, delta=0.1, width=0.1, plot_flag=True, zoomed=False)

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

ASYM = np.trapezoid(A_list_LR, x=angle_list) / np.trapezoid(A_list_RL, x=angle_list)
print(ASYM)

# %% ############################################################################################

data = {
    "T": T_list_LR, "A_RL": A_list_RL, "A_LR": A_list_LR,
    "R_RL": R_list_RL, "R_LR": R_list_LR, "T_bulk": trans_bulk,
    "A_bulk": emiss_bulk
}

tmm_h.plot_tra_curves(
    angle_list,
    data=data,
    xlabel='angle of incidence (degrees)',
    ncol_legend=2
)

tmm_h.show_textbox(text=f'n$_b$={nb}\nA={A}\nx$_0$={gam}\nASYM={np.round(ASYM, 2)}')

# %%
