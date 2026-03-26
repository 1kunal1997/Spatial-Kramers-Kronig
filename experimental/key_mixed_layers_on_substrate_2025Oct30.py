# script to test a few key Bruggeman mixture layers, on top of a 300um ZnS substrate. Plots presented Oct 30, 2025.

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

def bruggeman_eps_single(nk1: complex, nk2: complex, f2: float) -> complex:
    """Return Bruggeman effective RI for one fraction f2."""
    f1 = 1.0 - f2
    e1 = nk1**2
    e2 = nk2**2
    Hb = e1*(3*f1 - 1) + e2*(3*f2 - 1)
    rad = np.sqrt(Hb*Hb + 8*e1*e2)
    eps_plus  = (Hb + rad) / 4.0
    eps_minus = (Hb - rad) / 4.0

    # pick physical branch
    candidates = [eps_plus, eps_minus]
    physical = [e for e in candidates if np.imag(e) >= 0]
    if not physical:
        return np.sqrt(max(candidates, key=lambda z: np.imag(z)))
    return np.sqrt(min(physical, key=lambda z: np.imag(z)))


# %%
conc = 0.269
#conc_ZnS = 0.203
#conc_sapphire = 0.355
lambda_list, n_graphite, k_graphite = np.loadtxt(os.path.join(_PROJECT_ROOT, "RI", "graphite_nk.txt"), unpack=True)
#nk_sapphire = 1.595 + 1j*0.00066
_, n_sapphire, k_sapphire = np.loadtxt(os.path.join(_PROJECT_ROOT, "RI", "sapphire_nk_2-5um.txt"), unpack=True)
#nk_ZnS = 2.27
nk_MgF2 = 2.27
nk_eff_vs_lambda = []

pol = 'p'
angle = 0
T_list_LR, T_list_RL, R_list_LR, R_list_RL, A_list_LR, A_list_RL = (np.zeros_like(lambda_list) for _ in range(6))

n_list = [2.27, 2.15 + 1j*0.47]
d_list = [300, 0.03]

# add semi-infinite air layers
d_list.append(np.inf)
d_list.insert(0, np.inf)
n_list.append(1)       
n_list.insert(0, 1)

c_list = ['i','i','c','i']

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]
c_list_reversed = c_list[::-1]

for i, wl in enumerate(lambda_list):
    print(wl)
    nk_graphite = n_graphite[i] + 1j*k_graphite[i]
    #nk_sapphire = n_sapphire[i] + k_sapphire[i]
    nk_eff = bruggeman_eps_single(nk_MgF2, nk_graphite, conc)
    nk_eff_vs_lambda.append(nk_eff)
    n_list[2] = nk_eff
    n_list_reversed[1] = nk_eff

    T_list_LR[i], R_list_LR[i], A_list_LR[i] = tmm_h.TRA_inc(n_list, d_list, c_list, lamb=wl, angle=angle, pol=pol)
    T_list_RL[i], R_list_RL[i], A_list_RL[i] = tmm_h.TRA_inc(n_list_reversed, d_list_reversed, c_list_reversed, lamb=wl,
    angle=angle, pol=pol)

print(lambda_list[99])
print(nk_eff_vs_lambda[99])

# %%

data = {
    "T": T_list_LR,
    'A_LR': A_list_LR,
    'A_RL': A_list_RL,
    'R_LR': R_list_LR,
    'R_RL': R_list_RL
}
tmm_h.plot_tra_curves(
    lambda_list,
    data=data,
    title=f'Dispersionless (RI @ 3$\mu$m)'
    #title='With Dispersion'
)

# %%
xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Refractive Index'
title = f'Refractive Index of Graphite'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,n_graphite,label=r'n',color=colors.blue,auto_scale=True)
plot(fig,ax,lambda_list,k_graphite,label=r'k',color=colors.red,auto_scale=True)

title = f'Refractive Index of Sapphire'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)
plot(fig,ax,lambda_list,n_sapphire,label=r'n',color=colors.blue,auto_scale=True)
plot(fig,ax,lambda_list,k_sapphire,label=r'k',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)


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

# %%
