# %%

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import matplotlib.pyplot as plt
import colors
import tmm
import os

# %%
#n_list, d_list = tmm_h.generate_n_and_d_coating_logistic(plot_flag=True, zoomed=False)
n_list, d_list = tmm_h.HT_help(k=2, alpha=1, sigma=0.1, plot_flag=True)
##################################################################################

# %%
degrees = np.pi/180
nkdata_sapphire = np.genfromtxt(os.path.join('RI', 'lam_um_T_K_Al2O3_no_ko_ne_ke.dat'))
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

##################################################################################
# %% wavelength sweep, bulk sapphire
pol = 'p'
angle = 60
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

c_list = ['i','i','i']

for i, wl in enumerate(lamdata_sapphire):
    n_list[1] = ndata_sapphire[i] + 1j*kdata_sapphire[i]
    #print(f'wl: {wl}, n: {n_list[1]}')
    th_f[i] = tmm.snell(1, n_list[1], angle*degrees)
    R_front[i] = tmm.interface_R(pol, 1, n_list[1], angle*degrees, th_f[i])
    T_list_LR[i], R_list_LR[i], A_list_LR[i] = tmm_h.TRA_inc(n_list, d_list, c_list, lamb=wl, angle=angle*degrees, pol=pol)

R_back = R_list_LR - R_front

##################################################################################
# %% plot wavelength dependence of bulk sapphire and print noise

xlabel = 'Wavelength (um)'; ylabel = 'Fraction of Power'
title = f'Bulk Sapphire, {pol}-pol, angle={angle}$^{{\circ}}$'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lamdata_sapphire[0],lamdata_sapphire[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,angle_list,T_list_LR,label='T',color=colors.blue,auto_scale=True)
plot(fig,ax,lamdata_sapphire,R_back,label='R$_{back}$',color=colors.green,auto_scale=True)
#plot(fig,ax,lamdata_sapphire,R_front, '--', label='R$_{front}$',color=colors.green,auto_scale=True)
plot(fig,ax,lamdata_sapphire,A_list_LR,label='A',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)

R_noise = np.trapezoid(R_back, x=lamdata_sapphire) / (lamdata_sapphire[-1] - lamdata_sapphire[0])
A_avg = np.trapezoid(A_list_LR, x=lamdata_sapphire) / (lamdata_sapphire[-1] - lamdata_sapphire[0])

print(f'R_back noise is: {R_noise}')
print(f'average absorption is: {A_avg}')

##################################################################################
# %% add HT coating in front of bulk sapphire

A = 1.89
gam = 0.1
nb = 1.395
k_prop = 1

pol = 's'
angle = 80
T_list_LR, T_list_RL, R_list_LR, R_list_RL, A_list_LR, A_list_RL = (np.zeros_like(lamdata_sapphire) for _ in range(6))

#n_list, d_list = tmm_h.generate_n_and_d_coating_ellips(gam, A, nb, delta=0.05, plot_flag=True, zoomed=False)

#n_list, d_list = tmm_h.generate_n_and_d_coating_logistic(k=100, plot_flag=True, zoomed=False)
n_list, d_list = tmm_h.HT_help(k=100, sigma=0.01, plot_flag=True)
#n_list, d_list = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, delta=0.1, plot_flag=True, zoomed=True)

#max_index = np.argmax(np.array(n_list).real)
#min_index = np.argmin(np.array(n_list).real)

#print(f"min index is: {min_index} and n_list here is: {n_list[min_index]}")
#print(f"max index is: {max_index} and n_list here is: {n_list[max_index]}")

#for i, n in enumerate(n_list):
#    print(n)

n_coating = n_list
#n_coating = [num.real + 1j*num.imag*k_prop for num in n_coating]
d_coating = d_list
#for i, n in enumerate(n_coating):
#    print(n)

n_window = ndata_sapphire[0] + 1j*kdata_sapphire[0]
d_window = 5000
c_total = []
for _ in range(len(n_coating)):
    c_total.append('c')
n_total = n_coating
n_total.insert(0, n_window)
d_total = d_coating
d_total.insert(0, d_window)
c_total.insert(0, 'i')

##################################################################################
# %% wavelength dep. with coating

th_f = np.empty(len(lamdata_sapphire), dtype=np.complex128)
R_front = np.empty(len(lamdata_sapphire), dtype=float)

# add semi-infinite air layers
d_total.append(np.inf)
d_total.insert(0, np.inf)
n_total.append(1)       
n_total.insert(0, 1)
c_total.append('i')
c_total.insert(0, 'i')

#for i, n in enumerate(n_total):
#   print(f'n is: {n}, d is: {d_total[i]}, c is: {c_total[i]}')

for i, wl in enumerate(lamdata_sapphire):
    n_total[1] = ndata_sapphire[i] + 1j*kdata_sapphire[i]
    #print(f'wl: {wl}, n: {n_total[1]}')
    th_f[i] = tmm.snell(1, n_total[1], angle*degrees)
    R_front[i] = tmm.interface_R(pol, 1, n_total[1], angle*degrees, th_f[i])
    T_list_LR[i], R_list_LR[i], A_list_LR[i] = tmm_h.TRA_inc(n_total, d_total, c_total, lamb=wl, angle=angle*degrees, pol=pol)

R_back = R_list_LR - R_front

##################################################################################
# %% plot wavelength dependence of coating + bulk sapphire and print noise

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
title = f'Coating with HT of Truncated sKK (angle={angle}$^{{\circ}}$)'
#title = f'Incoherent GRIN (angle={angle}$^{{\circ}}$)'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lamdata_sapphire[0],lamdata_sapphire[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,lamdata_sapphire,T_list_LR,label='T',color=colors.blue,auto_scale=True)
plot(fig,ax,lamdata_sapphire,R_back,label='R$_{back}$',color=colors.green,auto_scale=True)
#plot(fig,ax,lamdata_sapphire,R_front, '--', label='R$_{front}$',color=colors.green,auto_scale=True)
#plot(fig,ax,lamdata_sapphire,R_list_LR, '*-', label='R$_{total}$',color=colors.green,auto_scale=True)
plot(fig,ax,lamdata_sapphire,A_list_LR,label='A',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)

R_noise = np.trapezoid(R_back, x=lamdata_sapphire) / (lamdata_sapphire[-1] - lamdata_sapphire[0])
A_avg = np.trapezoid(A_list_LR, x=lamdata_sapphire) / (lamdata_sapphire[-1] - lamdata_sapphire[0])

print(f'R_back noise is: {R_noise}')
print(f'average absorption is: {A_avg}')

##################################################################################
# %% angle sweep, bulk sapphire

pol = 'p'
lamb = 3
angle_list = np.arange(0,90,1)
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

c_list = ['i','i','i']

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle_inc(n_list, d_list, c_list, angle_list*degrees, lamb=lamb, pol=pol)
R_back = R_list_LR - R_front

##################################################################################
# %% 

xlabel = 'Angle (degrees)'; ylabel = 'Fraction of Power'
title = f'{pol}-pol, $\lambda$={lamb}$\mu$m'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(angle_list[0],angle_list[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,angle_list,T_list_LR,label='T',color=colors.blue,auto_scale=True)
plot(fig,ax,angle_list,R_back,label='R$_{back}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,R_front, '--', label='R$_{front}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,A_list_LR,label='A',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)

R_noise = np.trapezoid(R_back, x=angle_list) / (angle_list[-1] - angle_list[0])
A_avg = np.trapezoid(A_list_LR, x=angle_list) / (angle_list[-1] - angle_list[0])

print(f'R_back noise is: {R_noise}')
print(f'average absorption is: {A_avg}')

##################################################################################
# %% coating included angle sweep

A = 1.89
gam = 0.1
nb = 1.395
k_prop = 0

pol = 's'
lamb = 3
angle_list = np.arange(0,90,1)
T_list_LR, T_list_RL, R_list_LR, R_list_RL, A_list_LR, A_list_RL = (np.zeros_like(angle_list) for _ in range(6))

#n_list, d_list = tmm_h.generate_n_and_d_coating_ellips(gam, A, nb, delta=0.05, plot_flag=True, zoomed=True)
#n_list, d_list = tmm_h.generate_n_and_d_coating_logistic(nb=2.3, k=10)
n_list, d_list = tmm_h.HT_help(k=100, alpha=1, plot_flag=True)

#n_list, d_list = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, delta=0.1, plot_flag=True, zoomed=True)

#max_index = np.argmax(np.array(n_list).real)
#min_index = np.argmin(np.array(n_list).real)

#print(f"min index is: {min_index} and n_list here is: {n_list[min_index]}")
#print(f"max index is: {max_index} and n_list here is: {n_list[max_index]}")

#for i, n in enumerate(n_list):
#    print(n)

n_coating = n_list
#n_coating = [num.real + 1j*num.imag*k_prop for num in n_coating]
d_coating = d_list
#for i, n in enumerate(n_coating):
#    print(n)

idx = np.where(lamdata_sapphire == lamb)[0][0]
print(idx)
n_window = ndata_sapphire[idx] + 1j*kdata_sapphire[idx]
d_window = 5000
c_total = []
for _ in range(len(n_coating)):
    c_total.append('c')
n_total = n_coating
n_total.insert(0, n_window)
d_total = d_coating
d_total.insert(0, d_window)
c_total.insert(0, 'i')

##################################################################################
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
c_total.append('i')
c_total.insert(0, 'i')

#for i, n in enumerate(n_total):
#   print(f'n is: {n}, d is: {d_total[i]}, c is: {c_total[i]}')

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle_inc(n_total, d_total, c_total, angle_list*degrees, lamb=lamb, pol=pol)
R_back = R_list_LR - R_front

##################################################################################
# %% 

xlabel = 'Angle (degrees)'; ylabel = 'Fraction of Power'
if k_prop == 1:
    title = f'{pol}-pol, $\lambda$={lamb}$\mu$m with Coating'
else:
    title = f'{pol}-pol, $\lambda$={lamb}$\mu$m with Lossless Coating'

fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(angle_list[0],angle_list[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,angle_list,T_list_LR,label='T',color=colors.blue,auto_scale=True)
plot(fig,ax,angle_list,R_back,label='R$_{back}$',color=colors.green,auto_scale=True)
#plot(fig,ax,angle_list,R_front, '--', label='R$_{front}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,A_list_LR,label='A',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)

R_noise = np.trapezoid(R_back, x=angle_list) / (angle_list[-1] - angle_list[0])
A_avg = np.trapezoid(A_list_LR, x=angle_list) / (angle_list[-1] - angle_list[0])

print(f'R_back is: {R_noise}')
print(f'average absorbance is: {A_avg}')

##################################################################################
# %% ##############################################################################################

n_list = [2.27, 2.17 + 1j*0.48]
d_list = [300, 0.02]

angle_list = np.linspace(0,80,200)
degrees = np.pi/180
lamb = 3
pol = 'p'

th_f = np.zeros_like(angle_list)
R_front = np.zeros_like(angle_list)

for i, theta in enumerate(angle_list*degrees):
    th_f[i] = tmm.snell(1, 1.7255, theta)
    R_front[i] = tmm.interface_R(pol, 1, 1.7255, theta, th_f[i])

# add semi-infinite air layers
d_list.append(np.inf)
d_list.insert(0, np.inf)
n_list.append(1)       
n_list.insert(0, 1)

c_list = ['i','i','c','i']

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]
c_list_reversed = c_list[::-1]

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle_inc(n_list, d_list, c_list, lamb=lamb, angle_list=angle_list*degrees, pol=pol)
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_angle_inc(n_list_reversed, d_list_reversed, c_list_reversed, lamb=lamb,
 angle_list=angle_list*degrees, pol=pol)

vw_list = []
for j, angle in enumerate(angle_list*degrees):
    vw_list.append(tmm.inc_tmm('p', n_list, d_list, c_list, angle, lamb)['VW_list'])

R_noise = R_list_LR - R_front

data = {
    "T": T_list_LR,
    'A_LR': A_list_LR,
    'A_RL': A_list_RL,
    'R_LR': R_list_LR,
    'R_RL': R_list_RL
}
tmm_h.plot_tra_curves(
    angle_list,
    data=data,
    xlabel='Angle (degrees)',
    title=f'{pol}-pol, $\lambda$={lamb}$\mu$m'
)
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
##################################################################################
# %% sweep angles and wavelengths for colorplot of R_back average and A average

pol = 's'
angle_list = np.arange(0,90,1)
T_list_LR, T_list_RL, R_list_LR, R_list_RL, A_list_LR, A_list_RL, R_front, R_back = (np.zeros((len(angle_list), len(lamdata_sapphire)), dtype=float) for _ in range(8))
th_f = np.zeros((len(angle_list), len(lamdata_sapphire)), dtype=np.complex128)

n_list, d_list = tmm_h.HT_help(k=100, delta=0.01, alpha=1, plot_flag=True)

n_coating = n_list
d_coating = d_list
#for i, n in enumerate(n_coating):
#    print(n)

n_window = ndata_sapphire[0] + 1j*kdata_sapphire[0]
d_window = 5000
c_total = []
for _ in range(len(n_coating)):
    c_total.append('c')
n_total = n_coating
n_total.insert(0, n_window)
d_total = d_coating
d_total.insert(0, d_window)
c_total.insert(0, 'i')

# add semi-infinite air layers
d_total.append(np.inf)
d_total.insert(0, np.inf)
n_total.append(1)       
n_total.insert(0, 1)
c_total.append('i')
c_total.insert(0, 'i')

for i, angle in enumerate(angle_list*degrees):
    print(angle)
    for j, wl in enumerate(lamdata_sapphire):
        n_total[1] = ndata_sapphire[j] + 1j*kdata_sapphire[j]
        #print(f'wl: {wl}, n: {n_total[1]}')
        th_f[i][j] = tmm.snell(1, n_total[1], angle)
        R_front[i][j] = tmm.interface_R(pol, 1, n_total[1], angle, th_f[i][j])
        T_list_LR[i][j], R_list_LR[i][j], A_list_LR[i][j] = tmm_h.TRA_inc(n_total, d_total, c_total, lamb=wl, angle=angle, pol=pol)
        R_back[i][j] = R_list_LR[i][j] - R_front[i][j]

# %%
np.save(f"Data/Rback_HTlogistic_delta=0.01_k=100_{pol}-pol_angle=0-90~1_lamdata_sapphire.npy", R_back)

# %%

plt.figure()
plt.imshow((R_back).T, interpolation='none', aspect='auto', origin='lower', extent=(0, 90, lamdata_sapphire[0], lamdata_sapphire[-1]))
plt.ylabel('Wavelength ($\mu$m)')
plt.xlabel('AoI (degrees)')
plt.title('R_back')
ax = plt.gca()
plt.colorbar()

##################################################################################
# %% same as above but for bulk sapphire
pol = 'p'
angle_list = np.arange(0,90,1)
T_list_LR, T_list_RL, R_list_LR, R_list_RL, A_list_LR, A_list_RL, R_front, R_back = (np.zeros((len(angle_list), len(lamdata_sapphire)), dtype=float) for _ in range(8))
th_f = np.zeros((len(angle_list), len(lamdata_sapphire)), dtype=np.complex128)

n_list = [1]
d_list = [5000]

# add semi-infinite air layers
d_list.append(np.inf)
d_list.insert(0, np.inf)
n_list.append(1)       
n_list.insert(0, 1)

c_list = ['i','i','i']

for i, angle in enumerate(angle_list*degrees):
    print(angle)
    for j, wl in enumerate(lamdata_sapphire):
        n_list[1] = ndata_sapphire[j] + 1j*kdata_sapphire[j]
        #print(f'wl: {wl}, n: {n_list[1]}')
        th_f[i][j] = tmm.snell(1, n_list[1], angle)
        R_front[i][j] = tmm.interface_R(pol, 1, n_list[1], angle, th_f[i][j])
        T_list_LR[i][j], R_list_LR[i][j], A_list_LR[i][j] = tmm_h.TRA_inc(n_list, d_list, c_list, lamb=wl, angle=angle, pol=pol)
        R_back[i][j] = R_list_LR[i][j] - R_front[i][j]

np.save("Data/Rback_sapphire_bulk_5mm_p-pol_angle=0-90~1_lamdata_sapphire.npy", R_back)

##################################################################################
# %%

pol = 's'
R_back_GRIN = np.load(f"Data/Rback_HTlogistic_alpha=0_k=2_{pol}-pol_angle=0-90~1_lamdata_sapphire.npy")
R_back_sKK = np.load(f"Data/Rback_HTlogistic_k=100_{pol}-pol_angle=0-90~1_lamdata_sapphire.npy")
R_back_bulk = np.load(f"Data/Rback_sapphire_bulk_5mm_{pol}-pol_angle=0-90~1_lamdata_sapphire.npy")
plt.figure()
plt.imshow((R_back_sKK).T, interpolation='none', norm='linear', aspect='auto', origin='lower', extent=(0, 90, lamdata_sapphire[0], lamdata_sapphire[-1]))
plt.ylabel('Wavelength ($\mu$m)')
plt.xlabel('AoI (degrees)')
#plt.title(f'R$_{{GRIN}}$ / R$_{{sKK}}$ {pol}-pol, GRIN 50x thicker')
plt.title(f'R$_{{sKK}}$ {pol}-pol')
ax = plt.gca()
plt.colorbar()

##################################################################################
# %% alpha sweep over wavelengths

alpha_list = np.arange(0,1,0.01)
R_noise, A_avg = (np.zeros_like(alpha_list) for _ in range(2))
pol = 'p'
angle = 80
T_list_LR, T_list_RL, R_list_LR, R_list_RL, A_list_LR, A_list_RL, R_front = (np.zeros_like(lamdata_sapphire) for _ in range(7))
th_f = np.empty(len(lamdata_sapphire), dtype=np.complex128)

for j, alpha in enumerate(alpha_list):
    print(alpha)
    n_list, d_list = tmm_h.HT_help(k=100, alpha=alpha, plot_flag=False)

    n_coating = n_list
    d_coating = d_list

    n_window = ndata_sapphire[0] + 1j*kdata_sapphire[0]
    d_window = 5000
    c_total = []
    for _ in range(len(n_coating)):
        c_total.append('c')
    n_total = n_coating
    n_total.insert(0, n_window)
    d_total = d_coating
    d_total.insert(0, d_window)
    c_total.insert(0, 'i')

    # add semi-infinite air layers
    d_total.append(np.inf)
    d_total.insert(0, np.inf)
    n_total.append(1)       
    n_total.insert(0, 1)
    c_total.append('i')
    c_total.insert(0, 'i')

    #for i, n in enumerate(n_total):
    #   print(f'n is: {n}, d is: {d_total[i]}, c is: {c_total[i]}')

    for i, wl in enumerate(lamdata_sapphire):
        n_total[1] = ndata_sapphire[i] + 1j*kdata_sapphire[i]
        #print(f'wl: {wl}, n: {n_total[1]}')
        th_f[i] = tmm.snell(1, n_total[1], angle*degrees)
        R_front[i] = tmm.interface_R(pol, 1, n_total[1], angle*degrees, th_f[i])
        T_list_LR[i], R_list_LR[i], A_list_LR[i] = tmm_h.TRA_inc(n_total, d_total, c_total, lamb=wl, angle=angle*degrees, pol=pol)

    R_back = R_list_LR - R_front

    R_noise[j] = np.trapezoid(R_back, x=lamdata_sapphire) / (lamdata_sapphire[-1] - lamdata_sapphire[0])
    A_avg[j] = np.trapezoid(A_list_LR, x=lamdata_sapphire) / (lamdata_sapphire[-1] - lamdata_sapphire[0])

##################################################################################
# %% plot wavelength dependence of coating + bulk sapphire and print noise

xlabel = 'alpha'; ylabel = 'Fraction of Power'
#title = f'Coating with HT of Truncated sKK (angle={angle}$^{{\circ}}$)'
title = f'Tune Magnitude of Losses ({pol}-pol, angle={angle}$^{{\circ}}$)'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(alpha_list[0],alpha_list[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,lamdata_sapphire,T_list_LR,label='T',color=colors.blue,auto_scale=True)
plot(fig,ax,alpha_list,R_noise,label='R$_{avg}$',color=colors.green,auto_scale=True)
#plot(fig,ax,lamdata_sapphire,R_front, '--', label='R$_{front}$',color=colors.green,auto_scale=True)
#plot(fig,ax,lamdata_sapphire,R_list_LR, '*-', label='R$_{total}$',color=colors.green,auto_scale=True)
plot(fig,ax,alpha_list,A_avg,label='A$_{avg}$',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)

##################################################################################
# %% alpha sweep over angles

alpha_list = np.arange(0,1,0.01)
R_noise, A_avg = (np.zeros_like(alpha_list) for _ in range(2))
pol = 's'
lamb = 3
angle_list = np.arange(0,90,1)
T_list_LR, T_list_RL, R_list_LR, R_list_RL, A_list_LR, A_list_RL, R_front = (np.zeros_like(angle_list) for _ in range(7))
th_f = np.empty(len(angle_list), dtype=np.complex128)

for j, alpha in enumerate(alpha_list):
    print(alpha)
    n_list, d_list = tmm_h.HT_help(k=100, alpha=alpha, plot_flag=False)

    n_coating = n_list
    d_coating = d_list

    idx = np.where(lamdata_sapphire == lamb)[0][0]
    n_window = ndata_sapphire[idx] + 1j*kdata_sapphire[idx]
    d_window = 5000
    c_total = []
    for _ in range(len(n_coating)):
        c_total.append('c')
    n_total = n_coating
    n_total.insert(0, n_window)
    d_total = d_coating
    d_total.insert(0, d_window)
    c_total.insert(0, 'i')

    for i, theta in enumerate(angle_list*degrees):
        th_f[i] = tmm.snell(1, n_window, theta)
        R_front[i] = tmm.interface_R(pol, 1, n_window, theta, th_f[i])

    # add semi-infinite air layers
    d_total.append(np.inf)
    d_total.insert(0, np.inf)
    n_total.append(1)       
    n_total.insert(0, 1)
    c_total.append('i')
    c_total.insert(0, 'i')

    #for i, n in enumerate(n_total):
    #   print(f'n is: {n}, d is: {d_total[i]}, c is: {c_total[i]}')

    T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle_inc(n_total, d_total, c_total, angle_list*degrees, lamb=lamb, pol=pol)
    R_back = R_list_LR - R_front
    R_noise[j] = np.trapezoid(R_back, x=angle_list) / (angle_list[-1] - angle_list[0])
    A_avg[j] = np.trapezoid(A_list_LR, x=angle_list) / (angle_list[-1] - angle_list[0])

##################################################################################
# %% 

xlabel = 'alpha'; ylabel = 'Fraction of Power'
title = f'{pol}-pol, $\lambda$={lamb}$\mu$m'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(alpha_list[0],alpha_list[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,angle_list,T_list_LR,label='T',color=colors.blue,auto_scale=True)
plot(fig,ax,alpha_list,R_noise,label='R$_{avg}$',color=colors.green,auto_scale=True)
#plot(fig,ax,angle_list,R_front, '--', label='R$_{front}$',color=colors.green,auto_scale=True)
plot(fig,ax,alpha_list,A_avg,label='A$_{avg}$',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)


