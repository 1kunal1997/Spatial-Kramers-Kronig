# wavelength sweep and angle sweep of 5mm bulk sapphire vs. sKK truncated coating vs. GRIN . R_back calculated for ellipsometry, but outdated. presented Dec 17, 2025.
# STATUS: reviewed 2025-03-25. Sections 1-4 kept for refactoring. Uses generate_n_and_d_v6_symmetry, TRA, TRA_angle, plot_tra_curves.

# %%

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors
import tmm

# %%
degrees = np.pi/180
nkdata_sapphire = np.genfromtxt(os.path.join(_PROJECT_ROOT, 'RI', 'lam_um_T_K_Al2O3_no_ko_ne_ke.dat'))
kdata_sapphire = nkdata_sapphire[50:351, 3]
ndata_sapphire = nkdata_sapphire[50:351, 2]
lamdata_sapphire = nkdata_sapphire[50:351, 0]

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Refractive Index'
title = f'Real Refractive Index of Sapphire'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lamdata_sapphire[0],lamdata_sapphire[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lamdata_sapphire,ndata_sapphire,label=r'n',color=colors.blue,auto_scale=True)

title = f'Imaginary Refractive Index of Sapphire'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lamdata_sapphire[0],lamdata_sapphire[-1]),figsize=(5,4),auto_scale=True)
plot(fig,ax,lamdata_sapphire,kdata_sapphire,label=r'k',color=colors.red,auto_scale=True)

# %% wavelength sweep, bulk sapphire
pol = 'p'
angle = 0
T_list_LR, T_list_RL, R_list_LR, R_list_RL, A_list_LR, A_list_RL = (np.zeros_like(lamdata_sapphire) for _ in range(6))
th_f = np.empty(len(lamdata_sapphire), dtype=np.complex128)
R_front = np.empty(len(lamdata_sapphire), dtype=float)

n_list = [1]
d_list = [5000]

# add semi-infinite air layers
d_list.append(np.inf)
d_list.insert(0, np.inf)
n_list.append(1)       
n_list.insert(0, 1)

for i, wl in enumerate(lamdata_sapphire):
    n_list[1] = ndata_sapphire[i] + 1j*kdata_sapphire[i]
    print(f'wl: {wl}, n: {n_list[1]}')
    th_f[i] = tmm.snell(1, n_list[1], angle)
    R_front[i] = tmm.interface_R(pol, 1, n_list[1], angle, th_f[i])
    T_list_LR[i], R_list_LR[i], A_list_LR[i] = tmm_h.TRA(n_list, d_list, lamb=wl, angle=angle*degrees, pol=pol)

# %% ##################################################################################

xlabel = 'Wavelength (um)'; ylabel = 'Fraction of Power'
title = f'Bulk Sapphire'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lamdata_sapphire[0],lamdata_sapphire[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,angle_list,T_list_LR,label='T',color=colors.blue,auto_scale=True)
plot(fig,ax,lamdata_sapphire,R_list_LR,label='R$_{total}$',color=colors.green,auto_scale=True)
plot(fig,ax,lamdata_sapphire,R_front, '--', label='R$_{front}$',color=colors.green,auto_scale=True)
plot(fig,ax,lamdata_sapphire,A_list_LR,label='A',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)

# %%
R_back = R_list_LR - R_front
R_noise = np.trapezoid(R_back, x=lamdata_sapphire) / (lamdata_sapphire[-1] - lamdata_sapphire[0])
E_noise = np.trapezoid(A_list_LR, x=lamdata_sapphire) / (lamdata_sapphire[-1] - lamdata_sapphire[0])

noise = R_noise + E_noise
print(f'R noise is: {R_noise}')
print(f'E noise is: {E_noise}')

print(f'total noise is: {noise}')

# %% wavelength sweep with coating

A = 1.89
gam = 0.01
nb = 1.395
k_prop = 1

pol = 's'
angle = 0
T_list_LR, T_list_RL, R_list_LR, R_list_RL, A_list_LR, A_list_RL = (np.zeros_like(lamdata_sapphire) for _ in range(6))

n_list, d_list = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, delta=0.1, plot_flag=True, zoomed=True)

max_index = np.argmax(np.array(n_list).real)
min_index = np.argmin(np.array(n_list).real)

print(f"min index is: {min_index} and n_list here is: {n_list[min_index]}")
print(f"max index is: {max_index} and n_list here is: {n_list[max_index]}")

#for i, n in enumerate(n_list):
#    print(n)

n_coating = n_list[max_index:min_index+1]
n_coating = [num.real + 1j*num.imag*k_prop for num in n_coating]
d_coating = d_list[max_index:min_index+1]
for i, n in enumerate(n_coating):
    print(n)

n_window = ndata_sapphire[0] + 1j*kdata_sapphire[0]
d_window = 5000
n_total = n_coating
n_total.insert(0, n_window)
d_total = d_coating
d_total.insert(0, d_window)

# %%

th_f = np.empty(len(lamdata_sapphire), dtype=np.complex128)
R_front = np.empty(len(lamdata_sapphire), dtype=float)

# add semi-infinite air layers
d_total.append(np.inf)
d_total.insert(0, np.inf)
n_total.append(1)
n_total.insert(0, 1)

for i, n in enumerate(n_total):
   print(f'n is: {n}, d is: {d_total[i]}')

for i, wl in enumerate(lamdata_sapphire):
    n_total[1] = ndata_sapphire[i] + 1j*kdata_sapphire[i]
    print(f'wl: {wl}, n: {n_total[1]}')
    th_f[i] = tmm.snell(1, n_total[1], angle)
    R_front[i] = tmm.interface_R(pol, 1, n_total[1], angle, th_f[i])
    T_list_LR[i], R_list_LR[i], A_list_LR[i] = tmm_h.TRA(n_total, d_total, lamb=wl, angle=angle*degrees, pol=pol)

R_back = R_list_LR - R_front
# %% ##################################################################################

xlabel = 'Wavelength (um)'; ylabel = 'Fraction of Power'
title = f'Original Truncated sKK Coating'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lamdata_sapphire[0],lamdata_sapphire[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,angle_list,T_list_LR,label='T',color=colors.blue,auto_scale=True)
plot(fig,ax,lamdata_sapphire,R_list_LR,label='R$_{back}$',color=colors.green,auto_scale=True)
plot(fig,ax,lamdata_sapphire,R_front, '--', label='R$_{front}$',color=colors.green,auto_scale=True)
plot(fig,ax,lamdata_sapphire,A_list_LR,label='A',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)

# %%

R_noise = np.trapezoid(R_back, x=lamdata_sapphire) / (lamdata_sapphire[-1] - lamdata_sapphire[0])
E_noise = np.trapezoid(A_list_LR, x=lamdata_sapphire) / (lamdata_sapphire[-1] - lamdata_sapphire[0])

noise = R_noise + E_noise
print(f'R noise is: {R_noise}')
print(f'E noise is: {E_noise}')

print(f'total noise is: {noise}')

# %%

data = {
    "T": T_list_LR,
    'A_LR': A_list_LR,
    'R_LR': R_list_LR
}
tmm_h.plot_tra_curves(
    lamdata_sapphire,
    data=data,
    title=f'Normal Incidence'
)

# %% angle sweep, bulk sapphire

pol = 's'
lamb = 5
angle_list = np.arange(0,80,1)
T_list_LR, T_list_RL, R_list_LR, R_list_RL, A_list_LR, A_list_RL = (np.zeros_like(angle_list) for _ in range(6))

idx = np.where(lamdata_sapphire == lamb)[0][0]
print(idx)
n_list = [ndata_sapphire[idx] + 1j*kdata_sapphire[idx]]
d_list = [5000]
print(f'lam: {lamdata_sapphire[idx]}')
print(n_list[0])

th_f = np.empty(len(angle_list), dtype=np.complex128)
R_front = np.empty(len(angle_list), dtype=float)

for i, theta in enumerate(angle_list*degrees):
    th_f[i] = tmm.snell(1, n_list[0], theta)
    R_front[i] = tmm.interface_R(pol, 1, n_list[0], theta, th_f[i])

# add semi-infinite air layers
d_list.append(np.inf)
d_list.insert(0, np.inf)
n_list.append(1)       
n_list.insert(0, 1)

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle(n_list, d_list, angle_list*degrees, lamb=lamb, pol=pol)

# %% ##################################################################################

xlabel = 'Angle (degrees)'; ylabel = 'Fraction of Power'
title = f'{pol}-pol, $\lambda$={lamb}$\mu$m'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(angle_list[0],angle_list[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,angle_list,T_list_LR,label='T',color=colors.blue,auto_scale=True)
plot(fig,ax,angle_list,R_list_LR,label='R$_{total}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,R_front, '--', label='R$_{front}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,A_list_LR,label='A',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)

# %%
R_back = R_list_LR - R_front
R_noise = np.trapezoid(R_back, x=angle_list) / (angle_list[-1] - angle_list[0])
E_noise = np.trapezoid(A_list_LR, x=angle_list) / (angle_list[-1] - angle_list[0])

noise = R_noise + E_noise
print(f'R noise is: {R_noise}')
print(f'E noise is: {E_noise}')

print(f'total noise is: {noise}')
# %%

data = {
    "T": T_list_LR,
    'A_LR': A_list_LR,
    'R_LR': R_list_LR,
    'R_RL':R_front
}
tmm_h.plot_tra_curves(
    angle_list,
    data=data,
    xlabel='Angle (degrees)',
    title=f'{pol}-pol, $\lambda$={lamb}$\mu$m'
)

# %% coating included angle sweep

A = 1.89
gam = 1
nb = 1.395

pol = 's'
lamb = 5
angle_list = np.arange(0,80,1)
T_list_LR, T_list_RL, R_list_LR, R_list_RL, A_list_LR, A_list_RL = (np.zeros_like(angle_list) for _ in range(6))

n_list, d_list = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, delta=0.1, plot_flag=False, zoomed=True)

max_index = np.argmax(np.array(n_list).real)
min_index = np.argmin(np.array(n_list).real)

print(f"min index is: {min_index} and n_list here is: {n_list[min_index]}")
print(f"max index is: {max_index} and n_list here is: {n_list[max_index]}")

#for i, n in enumerate(n_list):
#    print(n)

n_coating = n_list[max_index:min_index+1]
n_coating = [num.real + 1j*num.imag*0.01 for num in n_coating]
d_coating = d_list[max_index:min_index+1]
for i, n in enumerate(n_coating):
    print(n)

idx = np.where(lamdata_sapphire == lamb)[0][0]
print(idx)
n_window = ndata_sapphire[idx] + 1j*kdata_sapphire[idx]
d_window = 5000
n_total = n_coating
n_total.insert(0, n_window)
d_total = d_coating
d_total.insert(0, d_window)

# %%

th_f = np.empty(len(angle_list), dtype=np.complex128)
R_front = np.empty(len(angle_list), dtype=float)

for i, theta in enumerate(angle_list*degrees):
    th_f[i] = tmm.snell(1, n_window, theta)
    R_front[i] = tmm.interface_R(pol, 1, n_window, theta, th_f[i])

# add semi-infinite air layers
d_total.append(np.inf)
d_total.insert(0, np.inf)
n_total.append(1)
n_total.insert(0, 1)

for i, n in enumerate(n_total):
   print(f'n is: {n}, d is: {d_total[i]}')

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle(n_total, d_total, angle_list*degrees, lamb=lamb, pol=pol)

# %% ##################################################################################

xlabel = 'Angle (degrees)'; ylabel = 'Fraction of Power'
title = f'{pol}-pol, $\lambda$={lamb}$\mu$m with Coating'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(angle_list[0],angle_list[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,angle_list,T_list_LR,label='T',color=colors.blue,auto_scale=True)
plot(fig,ax,angle_list,R_list_LR,label='R$_{total}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,R_front, '--', label='R$_{front}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,A_list_LR,label='A',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)

# %%
R_back = R_list_LR - R_front
R_noise = np.trapezoid(R_back, x=angle_list) / (angle_list[-1] - angle_list[0])
E_noise = np.trapezoid(A_list_LR, x=angle_list) / (angle_list[-1] - angle_list[0])

noise = R_noise + E_noise
print(f'R noise is: {R_noise}')
print(f'E noise is: {E_noise}')
print(f'total noise is: {noise}')

# %%
