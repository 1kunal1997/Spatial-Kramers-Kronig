# %%

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors
# %% ##############################################################################################
A = 3
gam = 0.01
nb = 1.7
delta = 0.4

n_list, d_list = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, delta=delta, plot_flag=True, zoomed=False)
print(f'Length of d_list: {len(d_list)}')
#n_list2, d_list2 = tmm_h.generate_n_and_d_v5_avg_over_cell(gam, A, nb, delta=delta, plot_flag=True, zoomed=True)
#print(len(d_list2))
#n_list3, d_list3 = tmm_h.generate_n_and_d_new(gam, A, nb, delta=delta, plot_flag=True, zoomed=True)
#print(len(d_list3))




# %% #############################################################################################

lambda_list = np.linspace(2,5,100)
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

# add semi-infinite air layers
d_list2.append(np.inf)
d_list2.insert(0, np.inf)
n_list2.append(nb)       
n_list2.insert(0, nb)

n_list_reversed2 = n_list2[::-1]
d_list_reversed2 = d_list2[::-1]

T_list_LR2, R_list_LR2, A_list_LR2 = tmm_h.TRA_wavelength(n_list2, d_list2, lambda_list)
T_list_RL2, R_list_RL2, A_list_RL2 = tmm_h.TRA_wavelength(n_list_reversed2, d_list_reversed2, lambda_list)

# add semi-infinite air layers
d_list3.append(np.inf)
d_list3.insert(0, np.inf)
n_list3.append(nb)       
n_list3.insert(0, nb)

n_list_reversed3 = n_list3[::-1]
d_list_reversed3 = d_list3[::-1]

T_list_LR3, R_list_LR3, A_list_LR3 = tmm_h.TRA_wavelength(n_list3, d_list3, lambda_list)
T_list_RL3, R_list_RL3, A_list_RL3 = tmm_h.TRA_wavelength(n_list_reversed3, d_list_reversed3, lambda_list)
# %% ############################################################################################

# plotting using Will's plot modules

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
title = f'TRA (A={A}, x$_0$={gam}$\mu$m)'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,T_list_LR,label=r'avg+symmetry',color=colors.blue,auto_scale=True)
plot(fig,ax,lambda_list,T_list_LR2, label=r'avg',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,T_list_LR3, label=r'og',color=colors.red,auto_scale=True)

#plot(fig,ax,lambda_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

#plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
#plot(fig,ax,lambda_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

#plot(fig,ax,lambda_list,T_list_RL2,label=r'T',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

#plot(fig,ax,lambda_list,A_list_RL2,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.light_red,auto_scale=True)
#plot(fig,ax,lambda_list,A_list_LR2, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.light_red,auto_scale=True)
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

#plot(fig,ax,lambda_list,R_list_RL2, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.light_green,auto_scale=True)
#plot(fig,ax,lambda_list,R_list_LR2, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.light_green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)
# %%
