# %%

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors

##################################################################################
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

##################################################################################
# %% constants

degrees = np.pi/180
conc = 0.25 #concentration of graphite
d_substrate = 300
n_substrate = 2.27 #ZnS
d_layer = 0.1
n_layer = 1 #placeholder for dispersive material. set in wavelength sweep
lambda_list, n_graphite, k_graphite = np.loadtxt("graphite_nk.txt", unpack=True)
_, n_sapphire, k_sapphire = np.loadtxt("sapphire_nk_2-5um.txt", unpack=True)
nk_eff = [] # effective RI of mixed layer using Bruggeman model
T_list_LR, T_list_RL, R_list_LR, R_list_RL, A_list_LR, A_list_RL = (np.zeros_like(lambda_list) for _ in range(6))

for i, wl in enumerate(lambda_list):
    nk_graphite = n_graphite[i] + 1j*k_graphite[i]
    nk_sapphire = n_sapphire[i] + 1j*k_sapphire[i]
    nk_eff.append(bruggeman_eps_single(nk_sapphire, nk_graphite, conc))

print(nk_eff[100])
##################################################################################
# %% plot RI of both materials and mixed material
xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Refractive Index'
title = f'Refractive Index of Graphite'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,n_graphite,label=r'n',color=colors.blue,auto_scale=True)
plot(fig,ax,lambda_list,k_graphite,label=r'k',color=colors.red,auto_scale=True)

title = f'Refractive Index of Sapphire'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)
plot(fig,ax,lambda_list,n_sapphire,label=r'n',color=colors.blue,auto_scale=True)
plot(fig,ax,lambda_list,k_sapphire,label=r'k',color=colors.red,auto_scale=True)

title = f'Refractive Index of Bruggeman Material'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)
plot(fig,ax,lambda_list,np.real(nk_eff),label=r'n',color=colors.blue,auto_scale=True)
plot(fig,ax,lambda_list,np.imag(nk_eff),label=r'k',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)

##################################################################################
# %% wavelength dependence of mixed layer

pol = 'p'
angle = 0

n_list = [n_substrate, n_layer]
d_list = [d_substrate, d_layer]

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
    n_list[2] = nk_eff[i]
    n_list_reversed[1] = nk_eff[i]

    T_list_LR[i], R_list_LR[i], A_list_LR[i] = tmm_h.TRA_inc(n_list, d_list, c_list, lamb=wl, angle=angle*degrees, pol=pol)
    T_list_RL[i], R_list_RL[i], A_list_RL[i] = tmm_h.TRA_inc(n_list_reversed, d_list_reversed, c_list_reversed, lamb=wl,
    angle=angle*degrees, pol=pol)

##################################################################################
# %% plot

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
    title=f'Bruggeman Mixed Layer, Normal Incidence',
    add_legend=False
)

##################################################################################
# %% angle dependence of mixed layer on substrate

angle_list = np.linspace(0,80,200)
lamb = 3
pol = 'p'

print(lambda_list[100])
print(nk_eff[100])
n_list = [n_substrate, nk_eff[100]]
d_list = [d_substrate, d_layer]

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

##################################################################################
# %% plot

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

##################################################################################
# %% alternate very thin layers of material A and B to see if it is equivalent to mixture
d_layer = 0.1
conc = 0.25
ratio = int(1/conc - 1)
N_pairs = 10
d_pairs = d_layer / N_pairs
d_graphite = conc*d_pairs
d_sapphire = d_graphite*ratio


print(ratio)
print(d_layer)
print(d_pairs)
print(d_graphite)
print(d_sapphire)


pol = 'p'
angle = 60

for i, wl in enumerate(lambda_list):
    n_list = [n_substrate]
    d_list = [d_substrate]
    c_list = ['i']

    for _ in range(N_pairs):
        """"""
        n_list.append(n_sapphire[i] + 1j*k_sapphire[i])
        d_list.append(d_sapphire) 
        c_list.append('c')
        
        n_list.append(n_graphite[i] + 1j*k_graphite[i])
        d_list.append(d_graphite)
        c_list.append('c')
        

    # add semi-infinite air layers
    d_list.append(np.inf)
    d_list.insert(0, np.inf)
    n_list.append(1)       
    n_list.insert(0, 1)
    c_list.append('i')
    c_list.insert(0,'i')

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]
    c_list_reversed = c_list[::-1]

    #for j,n in enumerate(n_list):
    #    print(f'n is: {n}, d is: {d_list[j]}, c is: {c_list[j]}')

    T_list_LR[i], R_list_LR[i], A_list_LR[i] = tmm_h.TRA_inc(n_list, d_list, c_list, lamb=wl, angle=angle*degrees, pol=pol)
    #print(f'Absorption at {wl} is {A_list_LR[i]}')
    T_list_RL[i], R_list_RL[i], A_list_RL[i] = tmm_h.TRA_inc(n_list_reversed, d_list_reversed, c_list_reversed, lamb=wl,
    angle=angle*degrees, pol=pol)
    #print(R_list_RL[i])

##################################################################################
# %% plot

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
    title=f'{N_pairs} stratified layers, {pol}-pol, angle={angle}$^{{\circ}}$',
    #title=f'{N_pairs} stratified layer, normal incidence',
    add_legend=False
)

##################################################################################
# %% lamellar EMT calculation

d_layer = 0.1
conc = 0.25

ng = n_graphite + 1j*k_graphite
ns = n_sapphire + 1j*k_sapphire

eps_g = ng**2
eps_s = ns**2

eps_par  = conc*eps_g + (1-conc)*eps_s
eps_perp = 1.0 / (conc/eps_g + (1-conc)/eps_s)

n_eff_par = np.sqrt(eps_par)
n_eff_perp = np.sqrt(eps_perp)

title = f'Refractive Index of Lamellar EMT Material'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)
plot(fig,ax,lambda_list,np.real(n_eff_par),label=r'n$_{\parallel}$',color=colors.blue,auto_scale=True)
plot(fig,ax,lambda_list,np.imag(n_eff_par),label=r'k$_{\parallel}$',color=colors.red,auto_scale=True)

plot(fig,ax,lambda_list,np.real(n_eff_perp), '--', label=r'n$_{\perp}$',color=colors.blue,auto_scale=True)
plot(fig,ax,lambda_list,np.imag(n_eff_perp), '--', label=r'k$_{\perp}$',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)

##################################################################################
# %% wavelength dependence of lamellar EMT

pol = 'p'
angle = 60

n_list = [n_substrate, n_layer]
d_list = [d_substrate, d_layer]

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
    n_list[2] = n_eff_par[i]
    n_list_reversed[1] = n_eff_par[i]

    T_list_LR[i], R_list_LR[i], A_list_LR[i] = tmm_h.TRA_inc(n_list, d_list, c_list, lamb=wl, angle=angle*degrees, pol=pol)
    T_list_RL[i], R_list_RL[i], A_list_RL[i] = tmm_h.TRA_inc(n_list_reversed, d_list_reversed, c_list_reversed, lamb=wl,
    angle=angle*degrees, pol=pol)

##################################################################################
# %% plot wavelength dependence

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
    title=f'Lamellar EMT layer, {pol}-pol, angle={angle}$^{{\circ}}$',
    #title=f'Lamellar EMT layer, normal incidence',
    add_legend=False
)

##################################################################################
# %% plot lamellar EMT vs. concentration

d_layer = 0.1
conc_arr = np.linspace(0,1,100)
print(conc_arr[-1])
n_eff_par = []
n_eff_perp = []
ng = n_graphite[100] + 1j*k_graphite[100]
ns = n_sapphire[100] + 1j*k_sapphire[100]

eps_g = ng**2
eps_s = ns**2

for i, conc in enumerate(conc_arr):
    eps_par  = conc*eps_g + (1-conc)*eps_s
    eps_perp = 1.0 / (conc/eps_g + (1-conc)/eps_s)

    n_eff_par.append(np.sqrt(eps_par))
    n_eff_perp.append(np.sqrt(eps_perp)) 

print(np.real(n_eff_par[-1]))
print(np.real(n_eff_perp[-1]))
xlabel = 'Concentration of Graphite'
title = f'Parallel Lamellar EMT Material'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(conc_arr[0],conc_arr[-1]),figsize=(5,4),auto_scale=True)
plot(fig,ax,conc_arr,np.real(n_eff_par),label=r'n',color=colors.blue,auto_scale=True)
plot(fig,ax,conc_arr,np.imag(n_eff_par),label=r'k',color=colors.red,auto_scale=True)

title = f'Perpendicular Lamellar EMT Material'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(conc_arr[0],conc_arr[-1]),figsize=(5,4),auto_scale=True)
plot(fig,ax,conc_arr,np.real(n_eff_perp),label=r'n',color=colors.blue,auto_scale=True)
plot(fig,ax,conc_arr,np.imag(n_eff_perp),label=r'k',color=colors.red,auto_scale=True)

# %%
