# %%

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors
# %% ##############################################################################################

A = 10
gam = 0.01
nb = 2.3

n_list, d_list = tmm_h.generate_n_and_d_new(gam, A, nb, delta=0.1, plot_flag=True, zoomed=True)
n_list, d_list = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, delta=0.1, plot_flag=True, zoomed=True)

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
# %% ############################################################################################

# plotting using Will's plot modules

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
title = f'TRA (A={A}, x$_0$={gam}$\mu$m)'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,lambda_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)

# %% #############################################################################################

data = {
    "T": T_list_LR,      # or T_list_RL, doesn't matter anymore
    "A_RL": A_list_RL,
    "A_LR": A_list_LR,
    "R_RL": R_list_RL,
    "R_LR": R_list_LR,
    "T_bulk": trans_bulk,
    "A_bulk": emiss_bulk
}

tmm_h.plot_tra_curves(
    lambda_list,
    data=data,
)

tmm_h.show_textbox(text=f'n$_b$={nb}\nA={A}\nx$_0$={gam}')

# %% #############################################################################################

gam = 0.01
A = 10
nb = 2.3
lamb = 3
angle_list = np.arange(0, 85, 1)
delta_angle = angle_list[-1] - angle_list[0]
angle_rad_list = angle_list*np.pi/180
a_prop_k = 0.8
a_prop_n = 0.5

n_list, d_list = tmm_h.generate_n_and_d(gam, A, nb, plot_flag=True)
n_list.imag = n_list.imag*a_prop_k
n_list.real = a_prop_n*n_list.real + (1-a_prop_n)*nb

losses_total = np.sum(d_list * np.imag(n_list))
trans_bulk = np.exp(-4*np.pi*losses_total/(lamb*np.cos(angle_rad_list)))
emiss_bulk = 1 - trans_bulk

# add semi-infinite air layers
d_list.append(np.inf)
d_list.insert(0, np.inf)
n_list.append(nb)       
n_list.insert(0, nb)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle(n_list, d_list, angle_rad_list, lamb=lamb, pol='s')
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_angle(n_list_reversed, d_list_reversed, angle_rad_list, lamb=lamb, pol='s')

A_avg = np.trapezoid(A_list_RL, x=angle_list) / delta_angle
T_avg = np.trapezoid(T_list_RL, x=angle_list) / delta_angle
FOM = T_avg / A_avg
print(f'FOM: {FOM}')
print(f'T_avg: {T_avg}')
print(f'A_avg: {A_avg}')

# %% ############################################################################################

# plotting using Will's plot modules

xlabel = 'Angle (degrees)'; ylabel = 'Fraction of Power'
#title = f'TRA (A={A}, x$_0$={gam}$\mu$m)'
title = 'TRA (smooth KK) p-pol'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(angle_list[0],angle_list[-1]),ylim=(-0.025,0.8),figsize=(5,4),auto_scale=True)

plot(fig,ax,angle_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,angle_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,angle_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,angle_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)
# %%
