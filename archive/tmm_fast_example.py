#%% 
import numpy as np
import torch
import tmm_fast as tmm_f
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
from tmm_fast.plotting_helper import plot_stacks

#%%

L = 12 # number of layers
d = np.random.uniform(20 , 150 , L)*1e-9 # thicknesses of the layers
d[0] = d[-1] = np.inf # set first and last layer as injection layer

n = np.random.uniform(1.2 , 5 , L) # random constant refractive index
n[-1] = 1 # outcoupling into air
wl = np.linspace(500 , 900 , 301)*1e-9
theta = np.deg2rad( np.linspace(0 , 90 , 301))
# here s and p polarization is computed and averaged to simulate incoherent light
result = (tmm_f.coh_tmm_fast('s', n , d , theta , wl )['R'] 
    + tmm_f.coh_tmm_fast('p', n , d , theta , wl )['R'])/2
# %%

fig, ax = plt.subplots(1,1)
print('1')
indexes = np.array([2, 1, 2.5, 1.6])
print('1')
thickness = np.array([5.0, 7, 3, 6])*1e-6
print('1')
labels = 'this is my stack'
print('1')
ax, cmap = plot_stacks(ax, indexes, thickness, labels=labels ) 
print('1')
plt.show()
print('1')

# %%

import numpy as np
from tmm_fast import coh_tmm as tmm

wl = np.linspace(400, 1200, 800) * (10**(-9))
theta = np.linspace(0, 45, 45) * (np.pi/180)
mode = 'T'
num_layers = 4
num_stacks = 128

#create m
M = np.ones((num_stacks, num_layers, wl.shape[0]))
for i in range(1, M.shape[1]-1):
    if np.mod(i, 2) == 1:
        M[:, i, :] *= 1.46
    else:
        M[:, i, :] *= 2.56

#create t
max_t = 150 * (10**(-9))
min_t = 10 * (10**(-9))
T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t
T[:, 0] = np.inf
T[:, -1] = np.inf

#tmm:
O = tmm('s', M, T, theta, wl, device='cpu')

print(':)')

# %%

import numpy as np
from tmm_fast import coh_tmm as tmm
import time

#wl = np.linspace(400, 1200, 800)
theta = np.linspace(0, 45, 45) * (np.pi/180)
wl = 650
#theta = 0
num_layers = 5
num_stacks = 1

'''
M = np.ones((num_layers, wl.shape[0]))
M[0, :] = 1
M[1, :] = 1.4
M[2, :] = 2.3
M[3, :] = 3
M[4, :] = 1
'''

M = np.ones((num_layers))
M[0] = 1
M[1] = 1.4
M[2] = 2.3
M[3] = 3
M[4] = 1

#create t
T = np.ones(num_layers)
T[1] = 50
T[2] = 100
T[3] = 20
T[0] = np.inf
T[-1] = np.inf

#tmm:
start = time.time()
O = tmm('s', M, T, theta, wl, device='cpu')
end = time.time()
print(end - start)
trans = O['T']
ref = O['R']
print(trans.shape)

# %%
import numpy as np
from matplotlib import pyplot as plt
from tmm_fast.plotting_helper import plot_stacks
fig, ax = plt.subplots(1,1)
indexes = np.array([2, 1, 2.5, 1.6])
thickness = np.array([5, 7, 3, 6])*1e-6
labels = 'this is my stack'
ax, cmap = plot_stacks(ax, indexes, thickness, labels=labels) 
plt.show()

# %%
from plot_functions import plot_setup, plot, legend
import colors
xlabel = 'Wavelength (um)'; ylabel = 'Fraction of Power'
title = f''
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(theta[0],theta[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,theta,trans,label='T$_{{avg}}$',color=colors.blue,auto_scale=True)
#plot(fig,ax,wl,A_LR_arr,label='A$_{{LR,avg}}$',linestyle='>-', markersize=8, markevery=15,color=colors.red,auto_scale=True)


# %%
import tmm_helper as tmm_h
import numpy as np
import time

n_list = [1, 1.4, 2.3, 3, 1]
d_list = [np.inf, 50, 100, 20, np.inf]
theta = np.linspace(0, 45, 45) * (np.pi/180)
wl = 650
start = time.time()
T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle(n_list, d_list, theta, lamb=wl, pol='s')
end = time.time()
print(end-start)
# %%

from plot_functions import plot_setup, plot, legend
import colors
xlabel = 'Wavelength (um)'; ylabel = 'Fraction of Power'
title = f''
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(theta[0],theta[-1]),figsize=(5,4),auto_scale=True)
#plot(fig,ax,theta,T_list_LR,label='T$_{{avg}}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,theta,trans, '--', label='T$_{{avg}}$',color=colors.red,auto_scale=True)
plot(fig,ax,theta,R_list_LR,label='T$_{{avg}}$',color=colors.green,auto_scale=True)
plot(fig,ax,theta,ref, '--', label='T$_{{avg}}$',color=colors.light_lavender,auto_scale=True)

# %%

import torch

a = torch.tensor([0.5+0.5j, 1.0+0.2j], dtype=torch.float64)
b = a.type(torch.complex128)

print(b)

# %%
