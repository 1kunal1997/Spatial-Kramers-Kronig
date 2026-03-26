#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import tmm

# Load external plotting functions
from plot_functions import plot_setup, plot, legend

# Load plotting colors
import colors # make available colors from schmid_colors.py 

# Image file settings
fmt = '.png' # image format (use png for PowerPoint, pdf and eps for publications)
dpi = 300 # image resolution, density of pixels per inch (use at least 300)

#%% ###########################################################################

trans_ZnS = np.loadtxt('ZnS--A.0.dpt')
n_ZnS_data = np.loadtxt('n_ZnS_exp.txt', skiprows=1)

'''
count = 0
while (n_ZnS_data[count,0] < 5):
    lamb = n_ZnS_data[count,0]
    if (lamb > 2):
        print(f'lamb is {lamb} at index {count}')
    count += 1
print (count)
print(n_ZnS_data[count,0])
'''
trans_ZnS = trans_ZnS[310:3109,:]
n_ZnS_data = n_ZnS_data[:196,:]

wls = np.linspace(2,5,100)
trans_ZnS_interp = interp1d(trans_ZnS[:,0], trans_ZnS[:,1], kind='linear', fill_value='extrapolate')
n_ZnS_interp = interp1d(n_ZnS_data[:,0], n_ZnS_data[:,1], kind='linear', fill_value='extrapolate')

trans_ZnS_new = trans_ZnS_interp(wls)
n_ZnS_new = n_ZnS_interp(wls)

#%% ################################################################################

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'transmittance'
title = "Transmittance Data"
fig,ax = plot_setup(xlabel,ylabel,xlim=(2,5),title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,trans_ZnS[:,0],trans_ZnS[:,1],color=colors.blue,auto_scale=True)
#plot(fig,ax,wls,trans_ZnS_new, '--', color=colors.red,auto_scale=True)

xlabel = 'Wavelength ($\mu$m)'; ylabel = 're(n)'
title = "Refractive Index Data"
fig,ax = plot_setup(xlabel,ylabel,xlim=(2,5),title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,n_ZnS_data[:,0],n_ZnS_data[:,1],color=colors.blue,auto_scale=True)
#plot(fig,ax,wls,n_ZnS_new, '--', color=colors.red,auto_scale=True)

#legend(fig,ax,auto_scale=True)

#%% ###################################################################################

m = -3e-7
b = 1.5e-6
d_list = [np.inf, 2000, np.inf]
c_list = ['i', 'i', 'i']
T_list = np.zeros_like(wls)
R_list = np.zeros_like(wls)
A_list = np.zeros_like(wls)
T_list2 = np.zeros_like(wls)
R_list2 = np.zeros_like(wls)
A_list2 = np.zeros_like(wls)
k_val = np.zeros_like(wls)

for i, lamb in enumerate(wls):

    k_val[i] = m*(lamb - 2) + b
    #k_val = 0
    n_list = [1, n_ZnS_new[i] + 1j*k_val[i], 1]
    n_list2 = [1, 2.26 + 1j*b, 1]

    T_list[i] = tmm.tmm_core.inc_tmm('s', n_list, d_list, c_list, 0, lamb)['T']
    R_list[i] = tmm.tmm_core.inc_tmm('s', n_list, d_list, c_list, 0, lamb)['R']
    A_list[i] = 1 - T_list[i] - R_list[i]

    T_list2[i] = tmm.tmm_core.inc_tmm('s', n_list2, d_list, c_list, 0, lamb)['T']
    R_list2[i] = tmm.tmm_core.inc_tmm('s', n_list2, d_list, c_list, 0, lamb)['R']
    A_list2[i] = 1 - T_list2[i] - R_list2[i]

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
title = "TRA of Window (no coating)"
fig,ax = plot_setup(xlabel,ylabel,xlim=(2,5),title=title,figsize=(5,4),auto_scale=True)

#plot(fig,ax,wls,trans_ZnS_new, color=colors.blue, label='data', auto_scale=True)
#plot(fig,ax,wls,T_list, color=colors.blue, label='T', auto_scale=True)
#plot(fig,ax,wls,R_list, color=colors.green, label='R', auto_scale=True)
#plot(fig,ax,wls,A_list, color=colors.red, label='A', auto_scale=True)
plot(fig,ax,wls,T_list2, color=colors.blue, label='T', auto_scale=True)
plot(fig,ax,wls,R_list2, color=colors.green, label='R', auto_scale=True)
plot(fig,ax,wls,A_list2, color=colors.red, label='A', auto_scale=True)

legend(fig,ax,auto_scale=True)

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'im(n)'
title = "Approximated Losses from Data"
fig,ax = plot_setup(xlabel,ylabel,xlim=(2,5),title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,wls,k_val, '--', color=colors.red, label='TMM', auto_scale=True)

# %%
