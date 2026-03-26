# %%

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors
import os
import matplotlib.pyplot as plt

# %% ##############################################################################################
def generate_n_and_d(gam, a, nb, delta=0.02, width_window=0, n_window=1, k_window=0, plot_flag=False, zoomed=True):
    
    dx      = gam/100               # Step size in 'continuous' Lorentzian
    xmax    = gam * 200           # Limits of Lorentzian

    nx      = 1 + int(np.floor(xmax / dx))
    xx      = np.linspace(0.0, xmax, nx)
    ee      = tmm_h.eps(xx,a,gam,nb)                    # Smooth Lorentzian curve
    #nk      = np.sqrt(ee)

    e_scale   = np.max(abs(ee-nb**2))
    x_scale = xmax

    count = 1
    xq, eq = [xx[0]], [ee[0]]
    for k in range(1, len(xx)):
        dx = (xx[k] - xq[-1]) / x_scale
        de = np.abs(ee[k] - eq[-1]) / e_scale
        ds = np.sqrt(dx**2 + de**2)
        
        if ds > delta:
            xq.append(xx[k])
            eq.append(ee[k])
            count = count + 1

    xq = np.append(xq,xmax) 
    eq = np.append(eq,ee[-1])
    count = count + 1
    #print(f'Number of Layers: {count}')

    '''
    for i, x in enumerate(xq):
        print(f'xq: {x}, imag(eps): {eq[i].imag}')
    print('\n')
    '''

    xq_sym = np.concatenate([-xq[:0:-1], xq])

    '''
    for i, x in enumerate(xq_sym):
        print(f'xq_sym: {x}')
    print('\n')
    '''

    d_list = np.diff(xq_sym)

    log_term = np.log(xq_sym[1:] + 1j*gam) - np.log(xq_sym[:-1] + 1j*gam)
    e_list = nb**2 - a*gam * log_term / d_list

    # add window
    e_window = (n_window + 1j*k_window)**2

    d_list_window = np.insert(d_list, len(d_list)//2, width_window)
    e_list_window = np.insert(e_list, len(d_list)//2, e_window)
    n_list_window = np.sqrt(e_list_window)

    # plot imaginary and real part of refractive index
    if (plot_flag):
        xx_sym = np.concatenate([-xx[:0:-1], xx])
        ee_sym     = tmm_h.eps(xx_sym,a,gam,nb)
        nk_sym      = np.sqrt(ee_sym)
        xq_sym_window = -0.5*np.sum(d_list_window) + np.concatenate(([0.0], np.cumsum(d_list_window)))
        tmm_h.nk_plot(xx_sym, ee_sym, nk_sym, xq_sym_window, n_list_window, gam, a, nb, zoomed)
        tmm_h.eps_plot(xx_sym, ee_sym, xq_sym_window, e_list_window, gam, a, nb, zoomed)
    
    return (n_list_window.tolist(), d_list_window.tolist())
# %% ##############################################################################################

# sample function call 
A = 2.2
gam = 0.2
A = 8.8
gam = 0.05
nb = 1.5

n_list, d_list = generate_n_and_d(gam, A, nb, delta=0.1, width_window=0.5, n_window=1.95, k_window=0.71, plot_flag=True, zoomed=True)
print(len(d_list))

# %% ############################################################################################

# import temperature dependent data, but wavelength only from 2-6 microns (losses very small)

nkdata_sapphire = np.genfromtxt(os.path.join('RI', 'lam_um_T_K_Al2O3_no_ko_ne_ke.dat'))
kdata_sapphire = nkdata_sapphire[50:451, 3]
ndata_sapphire = nkdata_sapphire[50:451, 2]
lamdata_sapphire = nkdata_sapphire[50:451, 0]

plt.figure()
#plt.plot(lamdata_sapphire, ndata_sapphire, label='n')
plt.plot(lamdata_sapphire, kdata_sapphire, label='k')
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Refractive Index')
plt.title('Refractive Index Data of Sapphire')

# %% ############################################################################################

# import temperature dependent data, but wavelength only from 2-6 microns (losses very small)

kdata_sapphire = np.genfromtxt('alumina_k_1~14um_Kischkat.txt', skip_header=1416, skip_footer=3, usecols=1)
ndata_sapphire = np.genfromtxt('alumina_n_1~14um_Kischkat.txt', skip_header=1416, skip_footer=3, usecols=1)
lamdata_sapphire = np.genfromtxt('alumina_n_1~14um_Kischkat.txt', skip_header=1416, skip_footer=3, usecols=0)

e_real_sapphire = ndata_sapphire**2 - kdata_sapphire**2
e_imag_sapphire = 2*ndata_sapphire*kdata_sapphire

for i, lam in enumerate(lamdata_sapphire):
    print(f'lam: {lam}, e_real: {e_real_sapphire[i]}, e_imag: {e_imag_sapphire[i]}')


plt.figure()
plt.plot(lamdata_sapphire, ndata_sapphire, label='n')
plt.plot(lamdata_sapphire, kdata_sapphire, label='k')
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Refractive Index')
plt.title('Refractive Index Data of Sapphire')
plt.legend(loc='upper right')

plt.figure()
plt.plot(lamdata_sapphire, e_real_sapphire, label='real')
plt.plot(lamdata_sapphire, e_imag_sapphire, label='imag')
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Dielectric Function')
plt.title('Dielectric Function of Sapphire')
plt.legend(loc='upper right')

xlabel = 'Wavelength ($\mu$m)'; ylabel = '$\epsilon^{\prime}$'; y2label = '$\epsilon^{\prime \prime}$'
title = f'Dielectric Function of Sapphire'
fig, ax, ax2 = plot_setup(xlabel,ylabel,True,y2label,title=title,yclr=colors.blue, y2clr=colors.red, figsize=(5,4),auto_scale=True, xlim=(lamdata_sapphire[0],lamdata_sapphire[-1]))

plot(fig,ax, lamdata_sapphire, e_real_sapphire, color=colors.blue,auto_scale=True)
plot(fig,ax2, lamdata_sapphire, e_imag_sapphire, color=colors.red,auto_scale=True)

# %% ############################################################################################

A = 2.5
gam = 0.2
nb = 1.13
delta = 0.1
width_window = 0.5
ASYM_avg = 0

T_LR, T_RL, R_LR, R_RL, A_LR, A_RL, emiss_bulk, trans_bulk = (np.empty(len(lamdata_sapphire), dtype=float) for i in range(8))
FOM_RL, FOM_LR, ASYM, FOM_bulk, FOM_enh = (np.empty(len(lamdata_sapphire), dtype=float) for i in range(5))

for i, lamb in enumerate(lamdata_sapphire):

    n_window = ndata_sapphire[i]
    k_window = kdata_sapphire[i]
    n_list, d_list = generate_n_and_d(gam, A, nb, delta=delta, width_window=width_window, n_window=n_window, k_window=k_window, plot_flag=False, zoomed=False)

    losses_total = np.sum(d_list * np.imag(n_list))
    trans_bulk[i] = np.exp(-4*np.pi*losses_total/lamb)
    emiss_bulk[i] = 1 - trans_bulk[i]

    # add semi-infinite air layers
    d_list.append(np.inf)
    d_list.insert(0, np.inf)
    n_list.append(nb)       
    n_list.insert(0, nb)

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]

    T_LR[i], R_LR[i], A_LR[i] = tmm_h.TRA(n_list, d_list, lamb)
    T_RL[i], R_RL[i], A_RL[i] = tmm_h.TRA(n_list_reversed, d_list_reversed, lamb)

    FOM_LR[i] = T_LR[i]**2 / A_LR[i]
    FOM_RL[i] = T_RL[i]**2 / A_RL[i]
    ASYM[i] = A_LR[i] / A_RL[i]
    FOM_bulk[i] = trans_bulk[i]**2 / emiss_bulk[i]
    FOM_enh[i] = FOM_RL[i] / FOM_bulk[i]

ASYM_avg = np.trapezoid(ASYM, lamdata_sapphire)/(lamdata_sapphire[-1]-lamdata_sapphire[0])
print(f'Average Asymmetry: {ASYM_avg}')
# %% ############################################################################################

data = {
    "T": T_LR, "A_RL": A_RL, "A_LR": A_LR, "R_RL": R_RL, "R_LR": R_LR,
    "T_bulk": trans_bulk, "A_bulk": emiss_bulk
}

tmm_h.plot_tra_curves(
    lamdata_sapphire,
    data=data,
    ncol_legend=2,
    title=f'n$_b$={nb}, A={A}, x$_0$={gam}$\mu m$, window={width_window}$\mu m$'
)

tmm_h.show_textbox(text=f'ASYM={np.round(ASYM_avg,2)}')

# %% ############################################################################################

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Emission Asymmetry'; y2label = 'Transmission'
title = f''
fig, ax, ax2 = plot_setup(xlabel,ylabel,True,y2label,title=title,yclr=colors.red, y2clr=colors.blue, figsize=(5,4),auto_scale=True, xlim=(lamdata_sapphire[0],lamdata_sapphire[-1]))

plot(fig,ax,lamdata_sapphire,ASYM,label=r'Asym.',color=colors.red,auto_scale=True)
plot(fig,ax2,lamdata_sapphire,T_LR,label=r'T',color=colors.blue,auto_scale=True)

# %% ##############################################################################################

A = 2.5
gam = 0.2
nb = 1.13
lamb = lamdata_sapphire[0]
degrees = np.pi/180
width_window = 0.5

n_list, d_list = generate_n_and_d(gam, A, nb, delta=0.1, width_window=width_window, n_window=ndata_sapphire[0], k_window=kdata_sapphire[0], plot_flag=True, zoomed=True)

angle_list = np.arange(0, 80, 1)
losses_total = np.sum(d_list * np.imag(n_list))
trans_bulk = np.exp(-4*np.pi*losses_total/(lamb*np.cos(angle_list*degrees)))
emiss_bulk = 1 - trans_bulk

# add semi-infinite air layers
d_list.append(np.inf)
d_list.insert(0, np.inf)
n_list.append(nb)       
n_list.insert(0, nb)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle(n_list, d_list, angle_list*degrees, lamb=lamb, pol='s')
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_angle(n_list_reversed, d_list_reversed, angle_list*degrees, lamb=lamb, pol='s')

ASYM = A_list_LR / A_list_RL
ASYM_avg = np.trapezoid(A_list_LR, x=angle_list) / np.trapezoid(A_list_RL, x=angle_list)
print(ASYM_avg)

# %% ############################################################################################

xlabel = 'Angle of Incidence (degrees)'; ylabel = 'Emission Asymmetry'; y2label = 'Transmission'
title = f''
fig, ax, ax2 = plot_setup(xlabel,ylabel,True,y2label,title=title,yclr=colors.red, y2clr=colors.blue, figsize=(5,4),auto_scale=True, xlim=(angle_list[0],angle_list[-1]))

plot(fig,ax,angle_list,ASYM,label=r'Asym.',color=colors.red,auto_scale=True)
plot(fig,ax2,angle_list,T_list_LR,label=r'T',color=colors.blue,auto_scale=True)
# %% ############################################################################################

data = {
    "T": T_list_LR, "A_RL": A_list_RL, "A_LR": A_list_LR,
    "R_RL": R_list_RL, "R_LR": R_list_LR, "T_bulk": trans_bulk,
    "A_bulk": emiss_bulk
}

tmm_h.plot_tra_curves(
    angle_list,
    data=data,
    xlabel='angle of incidence (degrees)',
    ncol_legend=2
)

tmm_h.show_textbox(text=f'n$_b$={nb}\nA={A}\nx$_0$={gam}\nASYM={np.round(ASYM, 2)}')

# %%
