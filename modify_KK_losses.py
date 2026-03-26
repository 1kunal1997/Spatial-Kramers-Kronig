# %%

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors
# %% ##############################################################################################

A = 10
gam = 0.05
nb = 2.3
lambda_list = np.linspace(2,5,100)
delta_lamb = lambda_list[-1] - lambda_list[0]

a_arr = np.arange(0, 1, 0.01)
R_LR_arr = np.zeros_like(a_arr)
R_RL_arr = np.zeros_like(a_arr)
T_arr = np.zeros_like(a_arr)
A_LR_arr = np.zeros_like(a_arr)
A_RL_arr = np.zeros_like(a_arr)

for i, a_prop_k in enumerate(a_arr):

    n_list, d_list = tmm_h.generate_n_and_d(gam, A, nb, a_prop_n=a_prop_k, plot_flag=False)
    losses_total = np.sum(d_list * np.imag(n_list))
    trans_bulk = np.exp(-4*np.pi*losses_total/lambda_list)
    emiss_bulk = 1 - trans_bulk

    # add semi-infinite air layers
    d_list.append(np.inf)
    d_list.insert(0, np.inf)
    n_list.append(nb)       
    n_list.insert(0, nb)

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]

    T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_wavelength(n_list, d_list, lambda_list)
    T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_wavelength(n_list_reversed, d_list_reversed, lambda_list)

    R_LR_arr[i] = np.trapz(R_list_LR, x=lambda_list) / delta_lamb
    R_RL_arr[i] = np.trapz(R_list_RL, x=lambda_list) / delta_lamb
    A_LR_arr[i] = np.trapz(A_list_LR, x=lambda_list) / delta_lamb
    A_RL_arr[i] = np.trapz(A_list_RL, x=lambda_list) / delta_lamb
    T_arr[i] = np.trapz(T_list_LR, x=lambda_list) / delta_lamb
# %% ############################################################################################

# plotting using Will's plot modules

xlabel = 'Proportion of Re(n) (n$_{{prop}}$)'; ylabel = 'Average over Wavelength'
title = f'Average TRA vs. Proportion of Re(n)'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(a_arr[0],a_arr[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,a_arr,T_arr,label='T$_{{avg}}$',color=colors.blue,auto_scale=True)
plot(fig,ax,a_arr,A_LR_arr,label='A$_{{LR,avg}}$',linestyle='>-', markersize=8, markevery=15,color=colors.red,auto_scale=True)
plot(fig,ax,a_arr,A_RL_arr,label='A$_{{RL,avg}}$',linestyle='<-', markersize=8, markevery=15,color=colors.red,auto_scale=True)
plot(fig,ax,a_arr,R_LR_arr,label='R$_{{LR,avg}}$',linestyle='>-', markersize=8, markevery=15,color=colors.green,auto_scale=True)
plot(fig,ax,a_arr,R_RL_arr,label='R$_{{RL,avg}}$',linestyle='<-', markersize=8, markevery=15,color=colors.green,auto_scale=True)

legend(fig,ax,auto_scale=True)
# %%
A = 10
gam = 0.05
nb = 2.3
lambda_list = np.linspace(2,5,100)

n_list, d_list = tmm_h.generate_n_and_d(gam, A, nb, a_prop_k=1, a_prop_n=0, plot_flag=True)
losses_total = np.sum(d_list * np.imag(n_list))
trans_bulk = np.exp(-4*np.pi*losses_total/lambda_list)
emiss_bulk = 1 - trans_bulk

# add semi-infinite air layers
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
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,lambda_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)
# %%
