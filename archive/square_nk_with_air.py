
#%% ##########################################################################################

import tmm

from numpy import pi, inf 
import numpy as np
import matplotlib.pyplot as plt

# Load external plotting functions
from plot_functions import plot_setup, plot, legend

# Load plotting colors
import colors # make available colors from schmid_colors.py 

# Image file settings
fmt = '.png' # image format (use png for PowerPoint, pdf and eps for publications)
dpi = 300 # image resolution, density of pixels per inch (use at least 300)
fig_dir = 'C:\\Users\\kl89\\MS Window Project\\Figures\\Squared Off n and k\\'

degree = pi/180

#%% #####################################################################################

lambda_list = np.linspace(2,5,50)
delta_lamb = lambda_list[-1] - lambda_list[0]
n_ZiS = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\RI\\ZiS_n_2~5um.txt')
k_ZiS = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\RI\\ZiS_k_2~5um.txt')
k_ZiS = np.interp(lambda_list, k_ZiS[:,0], k_ZiS[:,1])


n_graphite = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\RI\\graphite_n_2~5um.txt')
k_graphite = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\RI\\graphite_k_2~5um.txt')
np.savetxt('C:\\Users\\kl89\\MS Window Project\\RI\\graphite_n_vs_lam_2~5um.txt', np.column_stack((lambda_list, n_graphite)))
np.savetxt('C:\\Users\\kl89\\MS Window Project\\RI\\graphite_k_vs_lam_2~5um.txt', np.column_stack((lambda_list, k_graphite)))
n_silica = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\RI\\silica_n_2~5um.txt')
k_silica = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\RI\\silica_k_2~5um.txt')

n_silicon = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\RI\\silicon_n_2~5um.txt')
k_silicon = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\RI\\silicon_k_2~5um.txt')

xlabel = 'Wavelength ($\mu$m)'; ylabel = 're(n)'
title = "Real Refractive Index of Selected Materials"
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list, n_graphite , label = 'graphite',auto_scale=True, color=colors.blue)
plot(fig,ax,lambda_list, n_silica , label = 'silica',auto_scale=True, color=colors.green)
plot(fig,ax,lambda_list, n_silicon , label = 'silicon',auto_scale=True, color=colors.red)
plot(fig,ax,lambda_list, n_ZiS , label = 'zinc sulfide',auto_scale=True, color=colors.purple)

legend(fig,ax,auto_scale=True)

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'im(n)'
title = "Imaginary Refractive Index of Selected Materials"
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,lambda_list, k_graphite , label = 'graphite',auto_scale=True, color=colors.blue)
plot(fig,ax,lambda_list, k_silica , label = "silica",auto_scale=True, color=colors.green)
plot(fig,ax,lambda_list, k_silicon , label = "silicon",auto_scale=True, color=colors.red)
plot(fig,ax,lambda_list, k_ZiS , label = 'zinc sulfide',auto_scale=True, color=colors.purple)

legend(fig,ax,auto_scale=True)

#%% ######################################################################################

# function to calculate TRA of given ref. index, thicknesses, and wavelength range

def TRA_func(n_list, d_list, lambda_list):
    pol = 'p'
    angle = 0
    T_list = np.zeros_like(lambda_list)
    R_list = np.zeros_like(lambda_list)
    A_list = np.zeros_like(lambda_list)
    
    for j, lamb in enumerate(lambda_list):
        T_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['T']
        R_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['R']
        A_list[j] = 1 - T_list[j] - R_list[j]

    return (T_list, R_list, A_list)

def TRA_func_single(n_list, d_list, lamb, angle, pol):

    T = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['T']
    R = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['R']
    A = 1 - T - R

    return (T, R, A)

def TRA_func_single_inc(n_list, d_list, c_list, lamb, angle, pol):

    T = tmm.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)['T']
    R = tmm.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)['R']
    A = 1 - T - R

    return (T, R, A)

#%% ######################################################################################
# angle dependence at lambda_list[13] (~2.8 um), lambda_list = np.arange(2,5,50)

angle_list = np.arange(0, 90, 1)
pol = 's'
T_list_LR = np.zeros_like(angle_list, dtype=float)
R_list_LR = np.zeros_like(angle_list, dtype=float)
A_list_LR = np.zeros_like(angle_list, dtype=float)
T_list_RL = np.zeros_like(angle_list, dtype=float)
R_list_RL = np.zeros_like(angle_list, dtype=float)
A_list_RL = np.zeros_like(angle_list, dtype=float)
trans_bulk = np.zeros_like(angle_list, dtype=float)
refl_bulk = np.zeros_like(angle_list, dtype=float)
emiss_bulk = np.zeros_like(angle_list, dtype=float)

c_list = ['i', 'i', 'c', 'c', 'c', 'c', 'i']
idx = 13
wl = lambda_list[idx]
nb = n_ZiS[idx]
n_up = n_silicon[idx]     # silicon
n_down = n_silica[idx]    # silica
n_losses = n_graphite[idx]    # graphite
k_losses = k_graphite[idx]
for i, angle in enumerate(angle_list):

    d_list = [0.2, 0.1, 0.02, 0.1, 0.2]
    n_list = [nb, n_up, n_losses + 1j*k_losses, n_down, nb]
    losses_total = np.sum(d_list * np.imag(n_list))
    k_avg = losses_total / np.sum(d_list)

    d_list.append(inf)
    d_list.insert(0, inf)
    n_list.append(1)
    n_list.insert(0, 1)

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]
    c_list_reversed = c_list[::-1]

    T_list_LR[i], R_list_LR[i], A_list_LR[i] = TRA_func_single_inc(n_list, d_list, c_list, wl, angle*degree, pol)
    T_list_RL[i], R_list_RL[i], A_list_RL[i] = TRA_func_single_inc(n_list_reversed, d_list_reversed, c_list_reversed, wl, angle*degree, pol)

    # equivalently lossy bulk with air background
    #d_bulk = [np.sum(d_list[1:-1])]
    #n_bulk = [nb + 1j*k_avg]
    d_bulk = [0.02]
    n_bulk = [n_losses + 1j*k_losses]

    d_bulk.append(inf)
    d_bulk.insert(0, inf)
    n_bulk.append(1)
    n_bulk.insert(0, 1)

    trans_bulk[i], refl_bulk[i], emiss_bulk[i] = TRA_func_single(n_bulk, d_bulk, wl, angle*degree, pol)

FOM_LR = T_list_LR / A_list_LR
FOM_RL = T_list_RL / A_list_RL
FOM_bulk = trans_bulk / emiss_bulk

xlabel = 'Angle (degrees)'; ylabel = 'Fraction of Power'
title = f"TRA vs. Angle of Incidence ($\lambda$ = 2.8$\mu$m, {pol}-pol)"
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(angle_list[0],angle_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,angle_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
plot(fig,ax,angle_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,angle_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,angle_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,angle_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,angle_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,refl_bulk, '--', label=r'R$_{bulk}$',color=colors.light_green,auto_scale=True)

#text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nASYM = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)

xlabel = 'Angle (degrees)'; ylabel = 'FOM'
title = f"FOM vs. Angle of Incidence ($\lambda$ = 2.8$\mu$m, {pol}-pol)"
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(angle_list[0],angle_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,angle_list,FOM_RL,'<-', markersize=8, markevery=15, label=r'FOM$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,angle_list,FOM_LR, '>-', markersize=8, markevery=15, label=r'FOM$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,angle_list,FOM_bulk, '--', label=r'FOM$_{bulk}$',color=colors.light_red,auto_scale=True)

legend(fig,ax,auto_scale=True)

#%% ######################################################################################
# sweep high n layer (silicon) at lambda_list[13] (~2.8 um), lambda_list = np.arange(2,5,50)

thickness_up = np.arange(0.16, 0.24, 0.001)

T_list_LR = np.zeros_like(thickness_up, dtype=float)
R_list_LR = np.zeros_like(thickness_up, dtype=float)
A_list_LR = np.zeros_like(thickness_up, dtype=float)
T_list_RL = np.zeros_like(thickness_up, dtype=float)
R_list_RL = np.zeros_like(thickness_up, dtype=float)
A_list_RL = np.zeros_like(thickness_up, dtype=float)
trans_bulk = np.zeros_like(thickness_up, dtype=float)
refl_bulk = np.zeros_like(thickness_up, dtype=float)
emiss_bulk = np.zeros_like(thickness_up, dtype=float)

c_list = ['i', 'i', 'c', 'c', 'c', 'c', 'i']
idx = 13
m = -3e-7
b = 1.5e-6
wl = lambda_list[idx]
print(wl)
for i, d in enumerate(thickness_up):
    nb = n_ZiS[idx]
    n_up = n_silicon[idx]     # silicon
    k_up = k_silicon[idx]
    n_down = n_silica[idx]    # silica
    k_down = k_silica[idx]
    n_losses = n_graphite[idx]    # graphite
    k_losses = k_graphite[idx]
    k_val = m*(wl - 2) + b
    d_list = [2000, 0.1, 0.02, 0.1, d]
    n_list = [nb + 1j*k_val, n_up + 1j*k_up, n_losses + 1j*k_losses, n_down + 1j*k_down, nb + 1j*k_val]

    losses_total = np.sum(d_list * np.imag(n_list))
    k_avg = losses_total / np.sum(d_list)

    d_list.append(inf)
    d_list.insert(0, inf)
    n_list.append(1)
    n_list.insert(0, 1)

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]
    c_list_reversed = c_list[::-1]

    T_list_LR[i], R_list_LR[i], A_list_LR[i] = TRA_func_single_inc(n_list, d_list, c_list, wl, 0, 'p')
    T_list_RL[i], R_list_RL[i], A_list_RL[i] = TRA_func_single_inc(n_list_reversed, d_list_reversed, c_list_reversed, wl, 0, 'p')
    # equivalently lossy bulk with air background
    #d_bulk = [np.sum(d_list[1:-1])]
    d_bulk = [0.02]
    n_bulk = [ n_losses + 1j*k_losses]

    d_bulk.append(inf)
    d_bulk.insert(0, inf)
    n_bulk.append(1)
    n_bulk.insert(0, 1)

    trans_bulk[i], refl_bulk[i], emiss_bulk[i] = TRA_func_single(n_bulk, d_bulk, wl, 0, 'p')

#FOM_LR = T_list_LR / A_list_LR * T_list_LR
#FOM_RL = T_list_RL / A_list_RL * T_list_RL
#FOM_bulk = trans_bulk / emiss_bulk * trans_bulk
thickness_up = thickness_up * 1000
xlabel = 'Length of ZnS (nm)'; ylabel = 'Fraction of Power'
title = "TRA vs. ZnS Layer ($\lambda$ = 2.8$\mu$m)"
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(thickness_up[0],thickness_up[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,thickness_up,T_list_LR,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,thickness_up,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,thickness_up,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,thickness_up,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
#plot(fig,ax,thickness_up,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)
#plot(fig,ax,thickness_up,A_list_RL + A_list_LR, label=r'A',color=colors.red,auto_scale=True)

plot(fig,ax,thickness_up,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,thickness_up,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)
#plot(fig,ax,thickness_up,refl_bulk, '--', label=r'R$_{bulk}$',color=colors.light_green,auto_scale=True)

#text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nASYM = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)

'''
xlabel = 'Length (um)'; ylabel = 'FOM'
title = "FOM vs. Length of Outer n layers ($\lambda$ = 2.8$\mu$m)"
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(thickness_up[0],thickness_up[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,thickness_up,FOM_RL,'<-', markersize=8, markevery=15, label=r'FOM$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,thickness_up,FOM_LR, '>-', markersize=8, markevery=15, label=r'FOM$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,thickness_up,FOM_bulk, '--', label=r'FOM$_{bulk}$',color=colors.light_red,auto_scale=True)

legend(fig,ax,auto_scale=True)
'''

#%% ######################################################################################
# manual n_list and d_list to inlcude air AND dispersion in squared off nk

T_list_LR = np.zeros_like(lambda_list)
R_list_LR = np.zeros_like(lambda_list)
A_list_LR = np.zeros_like(lambda_list)
T_list_RL = np.zeros_like(lambda_list)
R_list_RL = np.zeros_like(lambda_list)
A_list_RL = np.zeros_like(lambda_list)
trans_bulk = np.zeros_like(lambda_list)
refl_bulk = np.zeros_like(lambda_list)
emiss_bulk = np.zeros_like(lambda_list)

fresnel_LR = np.zeros_like(lambda_list)
fresnel_RL = np.zeros_like(lambda_list)

m = -3e-7
b = 1.5e-6
c_list = ['i', 'i', 'c', 'c', 'c', 'c', 'i']
for i, wl in enumerate(lambda_list):
    nb = n_ZiS[i]
    kb = k_ZiS[i]
    n_up = n_silicon[i]     # silicon
    k_up = k_silicon[i]
    n_down = n_silica[i]    # silica
    k_down = k_silica[i]
    n_losses = n_graphite[i]    # graphite
    k_losses = k_graphite[i]

    k_val = m*(wl - 2) + b
    d_list = [2000, 0.2, 0.02, 0.1, 0.2]
    #d_list = [100, 0.55, 0.02, 0.2, 0.2]
    #n_list = [nb + 1j*kb, n_up + 1j*k_up, n_losses + 1j*k_losses, n_down + 1j*k_down, nb + 1j*kb]
    n_list = [nb + 1j*k_val, n_up + 1j*k_up, n_losses + 1j*k_losses, n_down + 1j*k_down, nb + 1j*k_val]
    #d_list = [1.92, 0.06, 0.02, 0.02, 0.06, 1.92]
    #n_list = [2.3, 3.32, 3.32 + 1j*1.87, 1.35+1.87*1j, 1.35, 2.3]
    #n_list = [2.3, 3.32, 3.81, 2.306, 1.35, 2.3]
    losses_total = np.sum(d_list * np.imag(n_list))
    #print(losses_total)
    k_avg = losses_total / np.sum(d_list)

    d_list.append(inf)
    d_list.insert(0, inf)
    n_list.append(1)
    n_list.insert(0, 1)

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]
    c_list_reversed = c_list[::-1]

    T_list_LR[i], R_list_LR[i], A_list_LR[i] = TRA_func_single_inc(n_list, d_list, c_list, wl, 0, 's')
    T_list_RL[i], R_list_RL[i], A_list_RL[i] = TRA_func_single_inc(n_list_reversed, d_list_reversed, c_list_reversed, wl, 0, 's')

    # equivalently lossy bulk with air background
    d_bulk = [0.02]
    n_bulk = [n_losses + 1j*k_losses]
    #d_bulk = [100]
    #n_bulk = [1.7 + 1j*0.00075]

    d_bulk.append(inf)
    d_bulk.insert(0, inf)
    n_bulk.append(1)
    n_bulk.insert(0, 1)

    trans_bulk[i], refl_bulk[i], emiss_bulk[i] = TRA_func_single(n_bulk, d_bulk, wl, 0, 's')

FOM_LR = T_list_LR / A_list_LR * T_list_LR
FOM_RL = T_list_RL / A_list_RL * T_list_RL
FOM_bulk = trans_bulk / emiss_bulk * trans_bulk

# plotting TRA

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
title = ""
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,lambda_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)
#plot(fig,ax,lambda_list,refl_bulk, '--', label=r'R$_{bulk}$',color=colors.light_green,auto_scale=True)

legend(fig,ax,auto_scale=True)

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'FOM'
title = "FOM"
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,FOM_RL,'<-', markersize=8, markevery=15, label=r'FOM$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,FOM_LR, '>-', markersize=8, markevery=15, label=r'FOM$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,FOM_bulk, '--', label=r'FOM$_{bulk}$',color=colors.light_red,auto_scale=True)

legend(fig,ax,auto_scale=True)


# %%

'''
nb = 2.3
n_up = 3.55     # silicon
n_down = 1.4    # silica
n_losses = 5    # graphite
k_losses = 4
'''
nb = n_ZiS[13]
n_up = n_silicon[13]     # silicon
n_down = n_silica[13]    # silica
n_losses = n_graphite[13]    # graphite
k_losses = k_graphite[13]

d_list = [0.4, 0.1, 0.02, 0.1, 0.2, 0.2]
n_list = [nb, n_up, n_losses + 1j*k_losses, n_down, nb, 1]
print(n_list)
x_list = np.cumsum([0] + d_list)

# Repeat x_list values and n_list values to create the step effect
x_plot = np.repeat(x_list, 2)[1:-1]
n_plot = np.repeat(n_list, 2)

xlabel = 'x ($\mu$m)'; ylabel = 'refractive index'
title = "Refractive Index of MS"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)
plot(fig,ax,x_plot, np.real(n_plot), label='re(n)',color=colors.blue,auto_scale=True)
plot(fig,ax,x_plot, np.imag(n_plot), '--', label='im(n)',color=colors.red,auto_scale=True)
legend(fig,ax,auto_scale=True)

losses_total = np.sum(d_list * np.imag(n_list))
print(losses_total)
k_avg = losses_total / np.sum(d_list)
print(k_avg)
d_list.append(inf)
d_list.insert(0, inf)
n_list.append(1)
n_list.insert(0, 1)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

T_list_LR, R_list_LR, A_list_LR = TRA_func(n_list, d_list, lambda_list)
T_list_RL, R_list_RL, A_list_RL = TRA_func(n_list_reversed, d_list_reversed, lambda_list)

# equivalently lossy bulk with air background
d_bulk = [np.sum(d_list[1:-1])]
n_bulk = [nb + 1j*k_avg]

d_bulk.append(inf)
d_bulk.insert(0, inf)
n_bulk.append(1)
n_bulk.insert(0, 1)

trans_bulk, refl_bulk, emiss_bulk = TRA_func(n_bulk, d_bulk, lambda_list)

FOM = np.trapz(T_list_RL, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list) * np.trapz(A_list_LR, x=lambda_list) / delta_lamb
ASYM = np.trapz(A_list_LR, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list)
T_avg = np.trapz(T_list_RL, x=lambda_list) / delta_lamb

# plotting TRA

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
title = "TRA for KK MS and Bulk"
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,lambda_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,refl_bulk, '--', label=r'R$_{bulk}$',color=colors.light_green,auto_scale=True)

#text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nASYM = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)

# %%
