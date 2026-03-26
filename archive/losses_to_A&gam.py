#%% ##########################################################################################

import tmm

from numpy import pi, inf 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from math import floor, log10
import time

# Load external plotting functions
from plot_functions import plot_setup, plot, legend

# Load plotting colors
import colors # make available colors from schmid_colors.py 

# Image file settings
fmt = '.png' # image format (use png for PowerPoint, pdf and eps for publications)
dpi = 300 # image resolution, density of pixels per inch (use at least 300)
fig_dir = 'C:\\Users\\kl89\\MS Window Project\\Figures\\Squared Off n and k\\'

degree = pi/180


#%% ##########################################################################################

# generate nk for spatial KK stack

def eps(x, a, gam, nb):
    return nb**2 - a * gam / (x + 1j*gam)

#%% ##########################################################################################

# plotting function for refractive index

def nk_plot(xx, nk, xq, n_list, gam, a, nb):
    xlabel = 'x ($\mu$m)'; ylabel = 'Re(n)'
    title = f''
    xlim = (xx[0]/10,xx[-1]/10)
    fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True, xlim=xlim)
    #fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)
    n_real = np.real(n_list)
    midpoints = (xq[:-1] + xq[1:]) / 2
    plot(fig,ax, xx, np.real(nk), label='smooth', color=colors.red,auto_scale=True)
    ax.stairs(n_real, xq, baseline=nb, label='discrete', linewidth = 2)
    plot(fig,ax, midpoints, n_real, '*', markersize=7, label='inputs', color=colors.green,auto_scale=True)
    text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, f"A = {a}\nx$_0$ = {gam}$\mu$m", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

    k_max   = max(np.max(np.imag(nk)), np.max(np.imag(n_list)))
    ylabel = 'Im(n)'
    title = f''
    fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True, xlim=xlim,ylim=(-k_max/20,k_max*(1+1/20)))

    n_imag = np.imag(n_list)
    plot(fig,ax, xx, np.imag(nk), label='smooth', color=colors.red,auto_scale=True)
    ax.stairs(n_imag, xq, baseline=0, label='discrete', linewidth = 2)
    plot(fig,ax, midpoints, n_imag, '*', markersize=7, label='inputs', color=colors.green,auto_scale=True)
    ax.text(0.05, 0.95, f"A = {a}\nx$_0$ = {gam}$\mu$m", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

#%% #############################################################################################

# function to generate list of refractive indices and thicknesses of each layer
# in TMM calculation

def generate_n_and_d(gam, a, nb, min_thickness=0.001, plot_flag=False):
    
    dx      = gam/100               # Step size in 'continuous' Lorentzian
    xmin    = -gam * 200           # Limits of Lorentzian
    xmax    = - xmin

    nx      = 1 + int(np.floor((xmax - xmin) / dx))
    xx      = np.linspace(xmin, xmax, nx)
    ee      = eps(xx,a,gam,nb)                    # Smooth Lorentzian curve
    nk      = np.sqrt(ee)

    k_max   = np.max(np.imag(nk))     # Max k value. used to set max n-step size 

    del_n   = k_max/200               # k_max/25. Max n-step size in discrete Lorentzian approximation
    del_x   = 10*gam                # 15*gam. Max x-step size in discrete Lorentzian approximation

    xq      = [xx[0]]                               
    nq      = [nk[0]]
    count   = 0

    for k in range(0,nx):
        if (abs((nk[k]) - (nq[count])) > del_n and abs((xx[k]) - (xq[count])) > min_thickness or 
            abs((xx[k]) - (xq[count])) > del_x):
            xq.append(xx[k])
            nq.append(nk[k])
            count = count + 1

    xq = np.append(xq,xmax)     # should we be appending xx[-1]? because xx does not include xmax as it is rn
    nq = np.append(nq,nk[-1])

    d_list = np.diff(xq)
    n_list = (nq[:-1] + nq[1:]) / 2

    
    # plot imaginary and real part of refractive index
    if (plot_flag):
        nk_plot(xx, nk, xq, n_list, gam, a, nb)
    
    return (n_list.tolist(), d_list.tolist())

#%% #############################################################################################

# function to generate list of refractive indices and thicknesses of each layer
# in TMM calculation

def generate_n_k_and_d(gam_n, gam_k, a_n, a_k, nb, min_thickness=0.001, plot_flag=False):
    
    dx      = gam_n/100               # Step size in 'continuous' Lorentzian
    xmin    = -gam_n * 200           # Limits of Lorentzian
    xmax    = - xmin

    nx      = 1 + int(np.floor((xmax - xmin) / dx))
    xx      = np.linspace(xmin, xmax, nx)
    ee_n    = eps(xx,a_n,gam_n,nb)                    # Smooth Lorentzian curve
    ee_k    = eps(xx,a_k,gam_k,nb)
    n_real  = np.sqrt(ee_n)
    n_imag  = np.sqrt(ee_k)

    k_max   = np.max(np.imag(n_real))     # Max k value. used to set max n-step size 

    del_n   = k_max/200               # k_max/25. Max n-step size in discrete Lorentzian approximation
    del_x   = 10*gam_n                # 15*gam. Max x-step size in discrete Lorentzian approximation

    xq      = [xx[0]]                               
    nq_real = [n_real[0]]
    nq_imag = [n_imag[0]]
    count   = 0

    for k in range(0,nx):
        if (abs((n_real[k]) - (nq_real[count])) > del_n and abs((xx[k]) - (xq[count])) > min_thickness or 
            abs((xx[k]) - (xq[count])) > del_x):
            xq.append(xx[k])
            nq_real.append(n_real[k])
            nq_imag.append(n_imag[k])
            #nq.append(np.real(n_real[k]) + 1j*np.imag(n_imag[k]))
            count = count + 1

    nq = np.real(nq_real) + 1j*np.imag(nq_imag)
    xq = np.append(xq,xmax)     # should we be appending xx[-1]? because xx does not include xmax as it is rn
    nq = np.append(nq,np.real(n_real[-1]) + 1j*np.imag(n_imag[-1]))

    d_list = np.diff(xq)
    n_list = (nq[:-1] + nq[1:]) / 2
    
    # plot imaginary and real part of refractive index
    if (plot_flag):
        nk_plot(xx, n_real, xq, n_list, gam_n, a_n, nb)
    
    return (n_list.tolist(), d_list.tolist())

#%% ######################################################################################
# trying new generate function

gam_n = 0.05
A_n = 10
gam_k = 1
A_k = 7
nb = 2.3
n_list, d_list = generate_n_k_and_d(gam_n, gam_k, A_n, A_k, nb, plot_flag=True)


# %% ##############################################################################
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
#%% ###############################################################################
table = {}
nb = 2.3
A_arr = np.arange(0.1, 10, 0.1)
gam_arr = np.arange(0.001, 1, 0.001)
start = time.time()
for A in A_arr:
    print(A)
    for gam in gam_arr:
        n_list, d_list = generate_n_and_d(gam, A, nb)
        losses_total = np.sum(d_list * np.imag(n_list))
        losses_rounded = '%s' % float('%.2g' % losses_total)
        table.setdefault(losses_rounded, []).append((A, gam))
end = time.time()
print(end - start)

# %% ############################################################################
for key, value in table.items():
    print(f"{key}: {value}")

# %% ###############################################################################
lambda_list = np.linspace(2,5,100)
delta_lamb = lambda_list[-1] - lambda_list[0]
losses = 1
losses = '%s' % float('%.2g' % losses)
pairs = table[losses]
T_avg_arr = []
ASYM_arr = []
A_arr = []
curr_pair = (0, 0)
for pair in pairs:
    if pair[0] != curr_pair[0] and pair[1] != curr_pair[1]:     #unique pair to avoid rounding errors
        curr_pair = pair
        n_list, d_list = generate_n_and_d(curr_pair[1], curr_pair[0], nb, plot_flag=False)
        A_arr.append(curr_pair[0])
        # add semi-infinite air layers
        d_list.append(inf)
        d_list.insert(0, inf)
        n_list.append(nb)            # change these to nb to see broadband zero-reflectance
        n_list.insert(0, nb)

        n_list_reversed = n_list[::-1]
        d_list_reversed = d_list[::-1]

        T_list_LR, R_list_LR, A_list_LR = TRA_func(n_list, d_list, lambda_list)
        T_list_RL, R_list_RL, A_list_RL = TRA_func(n_list_reversed, d_list_reversed, lambda_list)

        FOM = np.trapz(T_list_RL, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list) * np.trapz(A_list_LR, x=lambda_list) / delta_lamb
        ASYM = np.trapz(A_list_LR, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list)
        T_avg = np.trapz(T_list_RL, x=lambda_list) / delta_lamb

        T_avg_arr.append(T_avg)
        ASYM_arr.append(ASYM)

        print(f'for {pair}, T_avg: {T_avg} and ASYM: {ASYM}')

# %% ################################################################################
xlabel = 'A'; ylabel = ''
title = f'Average Transmittance with Losses = {losses}'
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)
plot(fig,ax, A_arr, T_avg_arr, color=colors.blue,auto_scale=True)

title = f'Emittance Asymmetry with Losses = {losses}'
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)
plot(fig,ax, A_arr, ASYM_arr, color=colors.red,auto_scale=True)

#%% ######################################################################################
# single TMM calculation for TRA, including plotting
gam_n = 0.05
A_n = 10
gam = 0.0015
A = 100
nb = 2.3
#n_list, d_list = generate_n_and_d(gam, A, nb, plot_flag=True)
n_list, d_list = generate_n_k_and_d(gam_n, gam, A_n, A, nb, plot_flag=True)
lambda_list = np.linspace(2,5,100)
losses_total = np.sum(d_list * np.imag(n_list))
print(losses_total)
trans_bulk = np.exp(-4*np.pi*losses_total/lambda_list)
emiss_bulk = 1 - trans_bulk

d_list.append(inf)
d_list.insert(0, inf)
n_list.append(nb)
n_list.insert(0, nb)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

delta_lamb = lambda_list[-1] - lambda_list[0]
T_list_LR, R_list_LR, A_list_LR = TRA_func(n_list, d_list, lambda_list)
T_list_RL, R_list_RL, A_list_RL = TRA_func(n_list_reversed, d_list_reversed, lambda_list)

FOM = np.trapz(T_list_RL, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list) * np.trapz(A_list_LR, x=lambda_list) / delta_lamb
ASYM = np.trapz(A_list_LR, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list)
T_avg = np.trapz(T_list_RL, x=lambda_list) / delta_lamb
print(f'T_avg: {T_avg}, ASYM: {ASYM}')
# plotting using Will's plot modules

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

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$ / A$_{{RL}}$ = {round(ASYM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)
#legend(fig,ax,auto_scale=True)
# %%
x = 0.00043
y = '%s' % float('%.1g' % x)
print(y)
# %%
print(np.logspace(-3, 0, num=100))
#%% ###############################################################################
dict2 = {}
nb = 2.3
gam_n = 0.05
A_n = 10
A_arr = np.logspace(-1, 2, num=100)
gam_arr = np.logspace(-3, 0, num=1000)
start = time.time()
for A in A_arr:
    print(A)
    for gam in gam_arr:
        #n_list, d_list = generate_n_k_and_d(gam_n, gam, A_n, A, nb)
        n_list, d_list = generate_n_and_d(gam, A, nb)
        losses_total = np.sum(d_list * np.imag(n_list))
        losses_rounded = '%s' % float('%.2g' % losses_total)
        dict2.setdefault(losses_rounded, []).append((A, gam))
end = time.time()
print(end - start)

# %% ###############################################################################
losses = 0.52
losses = '%s' % float('%.2g' % losses)
#print(dict2[losses])
for val in dict2[losses]:
    print(val)
# %%
lambda_list = np.linspace(2,5,100)
delta_lamb = lambda_list[-1] - lambda_list[0]
losses = 0.52
losses = '%s' % float('%.2g' % losses)
pairs = dict2[losses]

T_avg_arr = []
ASYM_arr = []
A_arr = []
gam_arr = []
curr_pair = (0, 0)
for pair in pairs:
    if pair[0] != curr_pair[0] and pair[1] != curr_pair[1]:     #unique pair to avoid rounding errors
        curr_pair = pair
        #n_list, d_list = generate_n_k_and_d(gam_n, curr_pair[1], A_n, curr_pair[0], nb, plot_flag=False)
        n_list, d_list = generate_n_and_d(curr_pair[1], curr_pair[0], nb, plot_flag=False)
        A_arr.append(curr_pair[0])
        gam_arr.append(curr_pair[1])
        # add semi-infinite air layers
        d_list.append(inf)
        d_list.insert(0, inf)
        n_list.append(nb)            # change these to nb to see broadband zero-reflectance
        n_list.insert(0, nb)

        n_list_reversed = n_list[::-1]
        d_list_reversed = d_list[::-1]

        T_list_LR, R_list_LR, A_list_LR = TRA_func(n_list, d_list, lambda_list)
        T_list_RL, R_list_RL, A_list_RL = TRA_func(n_list_reversed, d_list_reversed, lambda_list)

        FOM = np.trapz(T_list_RL, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list) * np.trapz(A_list_LR, x=lambda_list) / delta_lamb
        ASYM = np.trapz(A_list_LR, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list)
        T_avg = np.trapz(T_list_RL, x=lambda_list) / delta_lamb

        T_avg_arr.append(T_avg)
        ASYM_arr.append(ASYM)

        print(f'for {pair}, T_avg: {T_avg} and ASYM: {ASYM}')

# %% ################################################################################
xlabel = 'Log of x$_0$ (um)'; ylabel = ''
title = f'Average Transmittance with Losses = {losses}'
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)
plot(fig,ax, np.log10(gam_arr), T_avg_arr, color=colors.blue,auto_scale=True)

title = f'Emittance Asymmetry with Losses = {losses}'
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)
plot(fig,ax, np.log10(gam_arr), ASYM_arr, color=colors.red,auto_scale=True)

ylabel = 'Log of A'
title = f'A vs. x$_0$ with Losses = {losses}'
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)
plot(fig,ax, np.log10(gam_arr), np.log10(A_arr), color=colors.blue,auto_scale=True)

# %%
