# %%

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors
# %% ##############################################################################################

A = 5
gam = 0.01
#nb = 2.3
nb = 1.7

n_list, d_list = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, delta=0.01, plot_flag=True, zoomed=False)
#n_list, d_list = tmm_h.generate_n_and_d_5_layers(gam, A, nb, plot_flag=False, zoomed=True)

#n_list = np.array(n_list).real + 1j*np.array(n_list).imag*10
#n_list = n_list.tolist()
# %% #############################################################################################

lambda_list = np.linspace(1.88,5,50)
losses_total = np.sum(d_list * np.imag(n_list))
trans_bulk = np.exp(-4*np.pi*losses_total/lambda_list)
emiss_bulk = 1 - trans_bulk
print(losses_total)

# add semi-infinite air layers
d_list.append(np.inf)
d_list.insert(0, np.inf)
#n_list.append(n_list[-1])       
#n_list.insert(0, n_list[0])
n_list.append(nb)
n_list.insert(0,nb)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

'''
for i, n in enumerate(n_list):
    print(f'n: {np.round(n.real, 2)}, k: {np.round(n.imag, 2)}, d: {np.round(d_list[i], 2)}')

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_wavelength(n_list, d_list, lambda_list)
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_wavelength(n_list_reversed, d_list_reversed, lambda_list)
'''
T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_wavelength(n_list, d_list, lambda_list)
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_wavelength(n_list_reversed, d_list_reversed, lambda_list)
print(lambda_list)
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
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)

for i, lamb in enumerate(lambda_list):
    print(f'lam is: {np.round(lamb,2)}, R_LR is: {np.round(R_list_LR[i],2)}, R_RL is: {np.round(R_list_RL[i],3)}, T is: {np.round(T_list_LR[i],3)}')

# %% #############################################################################################

'''
data = {
    "T": T_list_LR,      # or T_list_RL, doesn't matter anymore
    "A_RL": A_list_RL,
    "A_LR": A_list_LR,
    "R_RL": R_list_RL,
    "R_LR": R_list_LR,
    "T_bulk": trans_bulk,
    "A_bulk": emiss_bulk
}
'''
data = {
    "T": T_list_LR,
    #"A_RL": A_list_RL,
    #'A_LR': A_list_LR,
    'R_LR': R_list_LR,
    'R_RL': R_list_RL
}
tmm_h.plot_tra_curves(
    lambda_list,
    data=data,
    title=''
)

tmm_h.show_textbox(text=f'n$_b$={nb}\nA={A}\nx$_0$={gam}')

print(T_list_LR[0])
print(A_list_LR[0])
print(A_list_RL[0])
print(R_list_RL[0])
print(R_list_LR[0])

# %% ########################################################

asym1 = np.trapezoid(A_list_LR - A_list_RL, x=lambda_list) / (lambda_list[-1] - lambda_list[0])

asym2 = np.trapezoid(A_list_LR / A_list_RL, x=lambda_list)

T_avg = np.trapezoid(T_list_LR, x=lambda_list) / (lambda_list[-1] - lambda_list[0])

print(asym1)
print(asym2)
print(T_avg)
# %% #############################################################################################

lamb = 3
pol = 's'
degrees = np.pi/180
angle_list = np.linspace(0,80,200)
angle_list_rad = angle_list*degrees
losses_total = np.sum(d_list * np.imag(n_list))
trans_bulk = np.exp(-4*np.pi*losses_total/lamb)
emiss_bulk = 1 - trans_bulk
print(losses_total)

# add semi-infinite air layers
d_list.append(np.inf)
d_list.insert(0, np.inf)
#n_list.append(n_list[-1].real)       
#n_list.insert(0, n_list[0].real)
n_list.append(nb)
n_list.insert(0,nb)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

'''
T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle(n_list, d_list, angle_list_rad, lamb=lamb, pol='s')
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_angle(n_list_reversed, d_list_reversed, angle_list_rad, lamb=lamb, pol='s')
'''

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle(n_list, d_list, angle_list_rad, lamb=lamb, pol=pol)
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_angle(n_list_reversed, d_list_reversed, angle_list_rad, lamb=lamb, pol=pol)

# %% ############################################################################################

# plotting using Will's plot modules

xlabel = 'Angle (degrees)'; ylabel = 'Fraction of Power'
title = f'TRA (A={A}, x$_0$={gam}$\mu$m)'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(angle_list[0],angle_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,angle_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,angle_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,angle_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
#plot(fig,ax,angle_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,angle_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)

# %% 

data = {
    "T": T_list_LR,
    #"A_RL": A_list_RL,
    #'A_LR': A_list_LR,
    'R_RL': R_list_RL,
    'R_LR': R_list_LR
}
tmm_h.plot_tra_curves(
    angle_list,
    data=data,
    xlabel='Angle (degrees)',
    title=f'lambda={lamb}$\mu m$, {pol}-pol',
    ylim=(0,1)
)
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

n_list, d_list = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, delta=0.01, plot_flag=True)
n_list = np.array(n_list)
n_list = n_list.real * a_prop_n + (1 - a_prop_n) * nb + 1j * n_list.imag * a_prop_k
n_list = n_list.tolist()

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
