# %%
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import tmm_helper as tmm_h
from plot_functions import plot_setup, plot, legend
import colors
degrees = np.pi/180
##################################################################################
# %%

files = ["silicon_n_0.25-1.45um_Schinke.txt",
         "silicon_k_0.25-1.45um_Schinke.txt",
         "gold_n_0.3-2.0um_thin_film_53nm_Yokubovsky.txt",
         "gold_k_0.3-2.0um_thin_film_53nm_Yokubovsky.txt",
         "sapphire_n_0.21-1.69um_Zhukovsky.txt",
         "graphite_n_0.03-10um.txt",
         "graphite_k_0.03-10um.txt"]  # <-- your 4 files (same 2 cols: lam[um]  n)
lambda_list = np.linspace(0.3, 1.45, 2000)                     # common lambda grid (um)
n = []
for f in files:
    d = np.loadtxt(f)
    n.append(interp1d(d[:,0], d[:,1], kind="linear")(lambda_list))
n_Si, k_Si, n_Au, k_Au, n_Al2O3, n_C, k_C = n

##################################################################################
# %%
plt.figure()
plt.plot(lambda_list, n_Si, label='Si')
plt.plot(lambda_list, n_Au, label='Au')
plt.plot(lambda_list, n_Al2O3, label='Al2O3')
plt.plot(lambda_list, n_C, label='C')
plt.legend()
plt.title('Real RI')
plt.ylabel('RI')
plt.xlabel('Wavelength ($\mu$m)')

plt.figure()
plt.plot(lambda_list, k_Si, label='Si')
plt.plot(lambda_list, k_Au, label='Au')
plt.plot(lambda_list, k_C, label='C')
plt.title('Imag RI')
plt.ylabel('RI')
plt.xlabel('Wavelength ($\mu$m)')

plt.legend()

##################################################################################
# %% alternate very thin layers of material A and B to see if it is equivalent to mixture

T_list_LR, A_list_LR, R_list_LR, Psi, Delta = (np.zeros_like(lambda_list) for _ in range(5))
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

pol = 's'
angle = 80

for i, wl in enumerate(lambda_list):
    n_list = []
    d_list = []
    c_list = []

    for _ in range(N_pairs):
        """"""
        n_list.append(n_Al2O3[i])
        d_list.append(d_sapphire) 
        c_list.append('c')
        
        n_list.append(n_C[i] + 1j*k_C[i])
        d_list.append(d_graphite)
        c_list.append('c')

    n_list.append(n_Au[i]+1j*k_Au[i])
    d_list.append(0.05)
    c_list.append('c')        

    # add semi-infinite air layers
    d_list.append(np.inf)
    d_list.insert(0, np.inf)
    n_list.append(n_Si[i]+1j*k_Si[i])       
    n_list.insert(0, 1)
    c_list.append('i')
    c_list.insert(0,'i')

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]
    c_list_reversed = c_list[::-1]

    #for j,n in enumerate(n_list):
    #    print(f'n is: {n}, d is: {d_list[j]}, c is: {c_list[j]}')

    T_list_LR[i], R_list_LR[i], A_list_LR[i] = tmm_h.TRA_inc(n_list, d_list, c_list, lamb=wl, angle=angle*degrees, pol=pol)
    Psi[i] = tmm_h.tmm.ellips(n_list, d_list, th_0=angle*degrees, lam_vac=wl)['psi']
    Delta[i] = tmm_h.tmm.ellips(n_list, d_list, th_0=angle*degrees, lam_vac=wl)['Delta']
    #print(f'Absorption at {wl} is {A_list_LR[i]}')
  
    #print(R_list_RL[i])

##################################################################################
# %%

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
title = f'{N_pairs} stratified layers, {pol}-pol, angle={angle}$^{{\circ}}$'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,T_list_LR,label='T',color=colors.blue,auto_scale=True)
plot(fig,ax,lambda_list,A_list_LR, label='A',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,R_list_LR, label='R',color=colors.green,auto_scale=True)

legend(fig,ax,auto_scale=True)

# %%

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'ellips parameters'
title = f'{N_pairs} stratified layers, angle={angle}$^{{\circ}}$'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,Psi,label='$\psi$',color=colors.blue,auto_scale=True)
plot(fig,ax,lambda_list,Delta, label='$\Delta$',color=colors.red,auto_scale=True)

legend(fig,ax,auto_scale=True)
##################################################################################
# %% lamellar EMT calculation

d_layer = 0.1
conc = 0.25

ng = n_C + 1j*k_C
ns = n_Al2O3

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

# %%
