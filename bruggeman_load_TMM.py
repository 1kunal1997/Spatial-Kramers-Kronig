# %%

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors

# %% ##############################################################################################

n_list = (np.loadtxt('nk_eff_graphite_only.txt', dtype=complex)).tolist()
d_list = (np.loadtxt('d_list_graphite_only.txt')).tolist()

print(n_list)
print(d_list)

angle_list = np.linspace(0,80,200)
degrees = np.pi/180
angle_list_rad = angle_list*degrees
lamb = 3
pol = 's'

# add semi-infinite air layers
d_list.append(np.inf)
d_list.insert(0, np.inf)
n_list.append(n_list[-1])       
n_list.insert(0, n_list[0])

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle(n_list, d_list, angle_list=angle_list_rad, pol=pol)
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_angle(n_list_reversed, d_list_reversed,angle_list=angle_list_rad, pol=pol) 

data = {
    "T": T_list_LR,
    "A_RL": A_list_RL,
    'A_LR': A_list_LR,
    'R_RL': R_list_RL,
    'R_LR': R_list_LR
}
tmm_h.plot_tra_curves(
    angle_list,
    data=data,
    xlabel='Angle (degrees)',
    title='5 Layer Bruggeman Model (s-pol)',
    ylim=(0,1)
)

# %% #############################################################################################
n_list = (np.loadtxt('nk_eff_graphite_only.txt', dtype=complex)).tolist()
d_list = (np.loadtxt('d_list_graphite_only.txt')).tolist()

lambda_list = np.linspace(2,5,100)
losses_total = np.sum(d_list * np.imag(n_list))
trans_bulk = np.exp(-4*np.pi*losses_total/lambda_list)
emiss_bulk = 1 - trans_bulk
print(losses_total)

# add semi-infinite air layers
d_list.append(np.inf)
d_list.insert(0, np.inf)
n_list.append(n_list[-1])       
n_list.insert(0, n_list[0])

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_wavelength(n_list, d_list, lambda_list)
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_wavelength(n_list_reversed, d_list_reversed, lambda_list)

# %% #############################################################################################

data = {
    "T": T_list_LR,
    "A_RL": A_list_RL,
    'A_LR': A_list_LR,
    'R_LR': R_list_LR,
    'R_RL': R_list_RL
}
tmm_h.plot_tra_curves(
    lambda_list,
    data=data,
    title='5 Layer Bruggeman Model',
    ylim=(-0.02,0.8)
)

# %% ########################################################

asym1 = np.trapezoid(A_list_LR - A_list_RL, x=lambda_list) / (lambda_list[-1] - lambda_list[0])

asym2 = np.trapezoid(A_list_LR / A_list_RL, x=lambda_list)
T_avg = np.trapezoid(T_list_LR, x=lambda_list) / (lambda_list[-1] - lambda_list[0])

print(asym1)
print(asym2)
print(T_avg)
# %% ############################################################################################

# plotting using Will's plot modules

xlabel = 'Angle (degrees)'; ylabel = 'Fraction of Power'
title = f''
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(angle_list[0],angle_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,angle_list,R_list_LR,label='R$_{total}$',color=colors.green,auto_scale=True)

plot(fig,ax,angle_list,R_front, '--', label='R$_{front}$',color=colors.green,auto_scale=True)

plot(fig,ax,angle_list,R_noise, '*-', markersize=8, markevery=15, label='R$_{noise}$',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)

# %%

noise = np.trapezoid(R_noise, x=angle_list) / (angle_list[-1] - angle_list[0])

print(f'total noise is: {noise}')

# %%
