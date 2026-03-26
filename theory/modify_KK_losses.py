# %%

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors
# %% ##############################################################################################
# Sweep the proportion of Re(n) deviation kept, while leaving Im(n) (losses) intact.
# At n_prop=0, Re(n)=nb everywhere (flat real index, only losses remain).
# At n_prop=1, full sKK profile is used.

A = 10
gam = 0.05
nb = 2.3
lambda_list = np.linspace(2, 5, 100)
delta_lamb = lambda_list[-1] - lambda_list[0]

n_prop_arr = np.arange(0, 1, 0.01)
R_LR_arr = np.zeros_like(n_prop_arr)
R_RL_arr = np.zeros_like(n_prop_arr)
T_arr = np.zeros_like(n_prop_arr)
A_LR_arr = np.zeros_like(n_prop_arr)
A_RL_arr = np.zeros_like(n_prop_arr)

# Generate the full sKK profile once
n_list_full, d_list_full = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, plot_flag=False)
n_arr_full = np.array(n_list_full)

for i, n_prop in enumerate(n_prop_arr):

    # Scale Re(n): interpolate between nb (flat) and full profile
    n_arr = n_prop * n_arr_full.real + (1 - n_prop) * nb + 1j * n_arr_full.imag
    n_list = n_arr.tolist()
    d_list = list(d_list_full)

    # Add semi-infinite air layers
    d_list.append(np.inf)
    d_list.insert(0, np.inf)
    n_list.append(nb)
    n_list.insert(0, nb)

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]

    T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_wavelength(n_list, d_list, lambda_list)
    T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_wavelength(n_list_reversed, d_list_reversed, lambda_list)

    R_LR_arr[i] = np.trapezoid(R_list_LR, x=lambda_list) / delta_lamb
    R_RL_arr[i] = np.trapezoid(R_list_RL, x=lambda_list) / delta_lamb
    A_LR_arr[i] = np.trapezoid(A_list_LR, x=lambda_list) / delta_lamb
    A_RL_arr[i] = np.trapezoid(A_list_RL, x=lambda_list) / delta_lamb
    T_arr[i] = np.trapezoid(T_list_LR, x=lambda_list) / delta_lamb
# %% ############################################################################################

xlabel = 'Proportion of Re(n) (n$_{{prop}}$)'; ylabel = 'Average over Wavelength'
title = f'Average TRA vs. Proportion of Re(n)'
fig, ax = plot_setup(xlabel, ylabel, title=title, xlim=(n_prop_arr[0], n_prop_arr[-1]), figsize=(5, 4), auto_scale=True)

plot(fig, ax, n_prop_arr, T_arr, label='T$_{{avg}}$', color=colors.blue, auto_scale=True)
plot(fig, ax, n_prop_arr, A_LR_arr, label='A$_{{LR,avg}}$', linestyle='>-', markersize=8, markevery=15, color=colors.red, auto_scale=True)
plot(fig, ax, n_prop_arr, A_RL_arr, label='A$_{{RL,avg}}$', linestyle='<-', markersize=8, markevery=15, color=colors.red, auto_scale=True)
plot(fig, ax, n_prop_arr, R_LR_arr, label='R$_{{LR,avg}}$', linestyle='>-', markersize=8, markevery=15, color=colors.green, auto_scale=True)
plot(fig, ax, n_prop_arr, R_RL_arr, label='R$_{{RL,avg}}$', linestyle='<-', markersize=8, markevery=15, color=colors.green, auto_scale=True)

legend(fig, ax, auto_scale=True)
# %% ############################################################################################
# Single run with n_prop=0 (flat Re(n)=nb, full losses): shows TRA spectrum

n_arr = nb + 1j * n_arr_full.imag
n_list = n_arr.tolist()
d_list = list(d_list_full)

d_list.append(np.inf)
d_list.insert(0, np.inf)
n_list.append(nb)
n_list.insert(0, nb)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_wavelength(n_list, d_list, lambda_list)
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_wavelength(n_list_reversed, d_list_reversed, lambda_list)

# %%

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
title = f'TRA for n_prop = 0'
fig, ax = plot_setup(xlabel, ylabel, title=title, xlim=(lambda_list[0], lambda_list[-1]), figsize=(5, 4), auto_scale=True)

plot(fig, ax, lambda_list, T_list_RL, label=r'T', color=colors.blue, auto_scale=True)
plot(fig, ax, lambda_list, A_list_RL, '<-', markersize=8, markevery=15, label=r'A$_{{RL}}$', color=colors.red, auto_scale=True)
plot(fig, ax, lambda_list, A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$', color=colors.red, auto_scale=True)
plot(fig, ax, lambda_list, R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$', color=colors.green, auto_scale=True)
plot(fig, ax, lambda_list, R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$', color=colors.green, auto_scale=True)

legend(fig, ax, auto_scale=True)
# %%
