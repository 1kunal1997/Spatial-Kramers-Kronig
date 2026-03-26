#!/usr/bin/env python
# coding: utf-8

#%% ##########################################################################################

import tmm

from numpy import pi, inf 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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

def nk_plot(xx, nk, xq, n_list, gam, a, nb, save_flag=False, min_thickness=0.01):
    xlabel = 'x ($\mu$m)'; ylabel = 'Re(n)'
    title = f''
    fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True, xlim=(xx[0]/10,xx[-1]/10))

    n_real = np.real(n_list)
    midpoints = (xq[:-1] + xq[1:]) / 2
    plot(fig,ax, xx, np.real(nk), label='smooth', color=colors.red,auto_scale=True)
    ax.stairs(n_real, xq, baseline=nb, label='discrete', linewidth = 2)
    plot(fig,ax, midpoints, n_real, '*', markersize=7, label='inputs', color=colors.green,auto_scale=True)
    text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, f"A = {a}\nx$_0$ = {gam}$\mu$m", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

    if save_flag:
        savename = f'Ref_index_real_nb~{nb}_A~{a}_gam~{gam}_squared_off_gam_n~8_shifted'
        plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)

    k_max   = max(np.max(np.imag(nk)), np.max(np.imag(n_list)))
    ylabel = 'Im(n)'
    title = f''
    fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True, xlim=(xx[0]/10,xx[-1]/10),ylim=(-k_max/20,k_max*(1+1/20)))

    n_imag = np.imag(n_list)
    plot(fig,ax, xx, np.imag(nk), label='smooth', color=colors.red,auto_scale=True)
    ax.stairs(n_imag, xq, baseline=0, label='discrete', linewidth = 2)
    plot(fig,ax, midpoints, n_imag, '*', markersize=7, label='inputs', color=colors.green,auto_scale=True)
    ax.text(0.05, 0.95, f"A = {a}\nx$_0$ = {gam}$\mu$m", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

    if save_flag:
        savename = f'Ref_index_imag_nb~{nb}_A~{a}_gam~{gam}_squared_off_gam_k~5_a_k~0.1'
        plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
        legend(fig,ax,auto_scale=True)
        plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)
#%% #############################################################################################

# function to generate list of refractive indices and thicknesses of each layer
# in TMM calculation

def generate_n_and_d(gam, a, nb, min_thickness=0.001, k_threshold_ratio=0, k_amplitude_ratio=1, plot_flag=True, save_flag=False, squared_off=False):
    
    dx      = gam/100               # Step size in 'continuous' Lorentzian
    xmin    = -gam * 200           # Limits of Lorentzian
    xmax    = - xmin

    nx      = 1 + int(np.floor((xmax - xmin) / dx))
    xx      = np.linspace(xmin, xmax, nx)
    ee      = eps(xx,a,gam,nb)                    # Smooth Lorentzian curve
    nk      = np.sqrt(ee)

    k_max   = np.max(np.imag(nk))     # Max k value. used to set max n-step size 

    # CHANGED FROM K_MAX/15 TO THIS!!!!!
    del_n   = k_max/50               # Max n-step size in discrete Lorentzian approximation
    del_x   = 5*gam                # Max x-step size in discrete Lorentzian approximation

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
    '''
    # make k below a certain value equal 0. set by 'k_threshold_ratio' 
    k_threshold = k_threshold_ratio*k_max
    for i, n_and_k in enumerate(n_list):
        if (np.imag(n_and_k) < k_threshold):
            n_list[i] = np.real(n_and_k)
        else:
            n_list[i] = np.real(n_and_k) + 1j*k_max/2

    n_threshold_top = (nb + np.max(np.real(nk)))/2
    n_threshold_bottom = (nb + np.min(np.real(nk)))/2
    for i, n_and_k in enumerate(n_list):
        if (np.real(n_and_k) < n_threshold_top and np.real(n_and_k) > n_threshold_bottom):
            n_list[i] = 1j*np.imag(n_and_k) + nb
        elif(np.real(n_and_k) > n_threshold_top):
            n_list[i] = 1j*np.imag(n_and_k) + n_threshold_top
        else:
            n_list[i] = 1j*np.imag(n_and_k) + n_threshold_bottom

    # change amplitude of k values by a fraction of original. set by 'k_amplitude_ratio' 
    n_list.imag = k_amplitude_ratio*n_list.imag
    '''
    # plot imaginary and real part of refractive index
    if (plot_flag):
        nk_plot(xx, nk, xq, n_list, gam, a, nb, min_thickness, save_flag)

    return (n_list.tolist(), d_list.tolist(), count)

def generate_square_n_and_d(gam, a, nb, a_prop_k=1, a_prop_n=1, gam_prop_k=1, gam_prop_n=1, n_kink = 0, plot_flag = True, save_flag=False):
    dx      = gam/100               # Step size in 'continuous' Lorentzian
    xmin    = -gam * 200           # Limits of Lorentzian
    xmax    = - xmin

    nx      = 1 + int(np.floor((xmax - xmin) / dx))
    xx      = np.linspace(xmin, xmax, nx)
    ee      = eps(xx,a,gam,nb)                    # Smooth Lorentzian curve
    nk      = np.sqrt(ee)

    xq      = [xx[0]]                               
    nq      = [nb]

    k_max   = np.max(np.imag(nk))
    n_max   = np.max(np.real(nk))
    n_min   = np.min(np.real(nk))
    print(k_max)
    print(n_max)
    print(n_min)
    k_val = a_prop_k*k_max
    n_val1 = a_prop_n*n_max + (1-a_prop_n)*nb
    n_val2 = a_prop_n*n_min + (1-a_prop_n)*nb
    if (gam_prop_n >= gam_prop_k):
        xq.append(-gam_prop_n*gam)
        nq.append(n_val1)

        xq.append(-gam_prop_k*gam)
        nq.append(n_val1 + n_kink + 1j*k_val)

        xq.append(0)
        nq.append(n_val2 + n_kink + 1j*k_val)

        xq.append(gam_prop_k*gam)
        nq.append(n_val2)

        xq.append(gam_prop_n*gam)
    else:
        xq.append(-gam_prop_k*gam)
        nq.append(nb + 1j*k_val)

        xq.append(-gam_prop_n*gam)
        nq.append(n_val1 + 1j*k_val)

        xq.append(0)
        nq.append(n_val2 + 1j*k_val)

        xq.append(gam_prop_n*gam)
        nq.append(nb + 1j*k_val)

        xq.append(gam_prop_k*gam)

    xq.append(xx[-1])
    nq.append(nb)

    xq = np.array(xq)

    d_list = np.diff(xq)
    n_list = np.array(nq)

    if plot_flag:
        nk_plot(xx, nk, xq, n_list, gam, a, nb, save_flag)

    return (n_list.tolist(), d_list.tolist())



# %%
gam = 0.01
A = 10
nb = 2.3

n_list, d_list = generate_square_n_and_d(gam, A, nb, gam_prop_n=8)
print(n_list)
print(d_list)


#%% #######################################################################################
# sample run of generate_n_and_d()

gam = 0.01
A = 10
nb = 2.3

generate_n_and_d(gam, A, nb, min_thickness=0.01);

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

def TRA_func_inc(n_list, d_list, c_list, lambda_list):
    pol = 'p'
    angle = 0
    T_list = np.zeros_like(lambda_list)
    R_list = np.zeros_like(lambda_list)
    A_list = np.zeros_like(lambda_list)
    
    for j, lamb in enumerate(lambda_list):
        T_list[j] = tmm.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)['T']
        R_list[j] = tmm.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)['R']
        A_list[j] = 1 - T_list[j] - R_list[j]

    return (T_list, R_list, A_list)

def TRA_func_angle(n_list, d_list, angle_list, lamb, pol='p'):
    T_list = np.zeros_like(angle_list)
    R_list = np.zeros_like(angle_list)
    A_list = np.zeros_like(angle_list)
    
    for j, angle in enumerate(angle_list):
        T_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['T']
        R_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['R']
        A_list[j] = 1 - T_list[j] - R_list[j]

    return (T_list, R_list, A_list)

def TRA_func_single(n_list, d_list, lamb):
    pol = 'p'
    angle = 0

    T = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['T']
    R = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['R']
    A = 1 - T - R

    return (T, R, A)

def TRA_func_more(n_list, d_list, lambda_list):
    pol = 'p'
    angle = 0
    print(len(n_list))
    T_list = np.zeros_like(lambda_list)
    R_list = np.zeros_like(lambda_list)
    A_list = np.zeros_like(lambda_list)
    vw_list = np.empty((len(lambda_list), len(n_list), 2), dtype=complex)
    kz_list = np.empty((len(lambda_list), len(n_list)), dtype=complex)
    theta_list = np.empty((len(lambda_list), len(n_list)), dtype=complex)
    
    for j, lamb in enumerate(lambda_list):
        T_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['T']
        R_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['R']
        A_list[j] = 1 - T_list[j] - R_list[j]
        vw_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['vw_list']
        kz_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['kz_list']
        theta_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['th_list']

    return (T_list, R_list, A_list, vw_list, kz_list, theta_list)

#%% ######################################################################################
# wavelength-sweep single TMM calculation for TRA, including plotting

gam = 0.01
A = 10
nb = 2.3
b = 1.5e-6
min_thickness=0.001
#n_list, d_list, num_layers = generate_n_and_d(gam, a, nb, min_thickness=min_thickness, k_threshold_ratio=0.5, save_flag=False)
n_list, d_list, count = generate_n_and_d(gam, A, nb, min_thickness=min_thickness)
c_list = ['c']
for i in range(count):
    c_list.append('c')
#print(n_list)
#print(d_list)
#print(c_list)
lambda_list = np.linspace(2,5,100)
losses_total = np.sum(d_list * np.imag(n_list))
print(losses_total)
trans_bulk = np.exp(-4*np.pi*losses_total/lambda_list)
emiss_bulk = 1 - trans_bulk

# add bulk window before spatial KK coating
d_list.insert(0, 2000)
n_list.insert(0, nb + 1j*b)
c_list.insert(0, 'i')

# add semi-infinite air layers
d_list.append(inf)
d_list.insert(0, inf)
n_list.append(1)            # change these to nb to see broadband zero-reflectance
n_list.insert(0, 1)
c_list.append('i')
c_list.insert(0, 'i')
print(n_list)
print(d_list)
print(c_list)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]
c_list_reversed = c_list[::-1]

delta_lamb = lambda_list[-1] - lambda_list[0]
T_list_LR, R_list_LR, A_list_LR = TRA_func_inc(n_list, d_list, c_list, lambda_list)
T_list_RL, R_list_RL, A_list_RL = TRA_func_inc(n_list_reversed, d_list_reversed, c_list_reversed, lambda_list)

FOM = np.trapz(T_list_RL, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list) * np.trapz(A_list_LR, x=lambda_list) / delta_lamb
ASYM = np.trapz(A_list_LR, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list)
T_avg = np.trapz(T_list_RL, x=lambda_list) / delta_lamb

# plotting using Will's plot modules

savename = f'TRA_nb={nb}_A={A}_x0={gam}_squared_off_gam_n~8_gam_k~5_a_k~0.1_shifted_n'
xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
title = f'TRA (A={A}, x$_0$={gam}$\mu$m)'
#title = "TRA (t$_n$ = 8x$_0$, t$_k$ = 5x$_0$, a$_k$ = 0.1A)"
#title = ""
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
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$ / A$_{{RL}}$ = {round(ASYM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

#plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
legend(fig,ax,auto_scale=True)
#plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

#%% ######################################################################################
# angle sweep single TMM calculation for TRA, including plotting

gam = 0.01
A = 10
nb = 2.3
lamb = 3
pol='s'
angle_list = np.arange(0, 85, 1)
delta_angle = angle_list[-1] - angle_list[0]
#min_thickness=0.01
#n_list, d_list, num_layers = generate_n_and_d(gam, a, nb, min_thickness=min_thickness, k_threshold_ratio=0.5, save_flag=False)
n_list, d_list = generate_square_n_and_d(gam, A, nb, gam_prop_n=8, gam_prop_k=2, save_flag=False)
print(n_list)
print(d_list)

n_list2, d_list2, count = generate_n_and_d(gam, A, nb)
'''
losses_total = np.sum(d_list * np.imag(n_list))
print(losses_total)
trans_bulk = np.exp(-4*np.pi*losses_total/lambda_list)
emiss_bulk = 1 - trans_bulk
'''
d_list.append(inf)
d_list.insert(0, inf)
n_list.append(nb)
n_list.insert(0, nb)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

d_list2.append(inf)
d_list2.insert(0, inf)
n_list2.append(nb)
n_list2.insert(0, nb)

n_list2_reversed = n_list2[::-1]
d_list2_reversed = d_list2[::-1]

T_list_LR, R_list_LR, A_list_LR = TRA_func_angle(n_list, d_list, angle_list*degree, lamb, pol=pol)
T_list_RL, R_list_RL, A_list_RL = TRA_func_angle(n_list_reversed, d_list_reversed, angle_list*degree, lamb, pol=pol)

T_list_LR2, R_list_LR2, A_list_LR2 = TRA_func_angle(n_list2, d_list2, angle_list*degree, lamb, pol=pol)
T_list_RL2, R_list_RL2, A_list_RL2 = TRA_func_angle(n_list2_reversed, d_list2_reversed, angle_list*degree, lamb, pol=pol)

A_avg = np.trapz(A_list_RL, x=angle_list) / delta_angle
T_avg = np.trapz(T_list_RL, x=angle_list) / delta_angle
FOM = T_avg / A_avg
print(f'FOM: {FOM}')
print(f'T_avg: {T_avg}')
print(f'A_avg: {A_avg}')

# %% #########################################################################################

# plotting using Will's plot modules

savename = f'TRA_nb={nb}_A={A}_x0={gam}_squared_off_gam_n~8_gam_k~5_a_k~0.1_shifted_n'
xlabel = 'Angle (degrees)'; ylabel = 'Fraction of Power'
#title = f'TRA (A={A}, x$_0$={gam}$\mu$m, t$_{{min}}$={min_thickness}$\mu$m)'
#title = "TRA (t$_n$ = 8x$_0$, t$_k$ = 5x$_0$, a$_k$ = 0.1A)"
title = "Reflectance (s-pol)"
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(angle_list[0],angle_list[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,angle_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

#plot(fig,ax,angle_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
#plot(fig,ax,angle_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

#plot(fig,ax,angle_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
#plot(fig,ax,angle_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,R_list_LR2, label=r'R$_{LR, smooth}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,R_list_LR, '--', label=r'R$_{LR, square}$',color=colors.green,auto_scale=True)



#text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nASYM = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$ / A$_{{RL}}$ = {round(ASYM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)
#plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
legend(fig,ax,auto_scale=True)
#plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

#%% ######################################################################################
# single TMM calculation for TRA, including plotting

gam = 0.01
A = 10
nb = 2.3
#min_thickness=0.01
#n_list, d_list, num_layers = generate_n_and_d(gam, a, nb, min_thickness=min_thickness, k_threshold_ratio=0.5, save_flag=False)
n_list, d_list = generate_square_n_and_d(gam, A, nb, gam_prop_n=8, gam_prop_k=2, save_flag=False)
print(n_list)
print(d_list)
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
T_list_LR, R_list_LR, A_list_LR, vw_list, kz_list, theta_list = TRA_func_more(n_list, d_list, lambda_list)
T_list_RL, R_list_RL, A_list_RL = TRA_func(n_list_reversed, d_list_reversed, lambda_list)

FOM = np.trapz(T_list_RL, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list) * np.trapz(A_list_LR, x=lambda_list) / delta_lamb
ASYM = np.trapz(A_list_LR, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list)
T_avg = np.trapz(T_list_RL, x=lambda_list) / delta_lamb

print(vw_list[0])
print(kz_list[0])
print(theta_list[0])
# plotting using Will's plot modules

savename = f'TRA_nb={nb}_A={A}_x0={gam}_squared_off_gam_n~8_gam_k~5_a_k~0.1_shifted_n'
xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
#title = f'TRA (A={A}, x$_0$={gam}$\mu$m, t$_{{min}}$={min_thickness}$\mu$m)'
#title = "TRA (t$_n$ = 8x$_0$, t$_k$ = 5x$_0$, a$_k$ = 0.1A)"
title = ""
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,lambda_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)


text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nASYM = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)
ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$ / A$_{{RL}}$ = {round(ASYM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)
#plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
legend(fig,ax,auto_scale=True)
#plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

#%% ######################################################################################
# manual n_list and d_list to inlcude air in squared off nk
nb = 2.3
n_up = 3.55     # silicon
n_down = 1.4    # silica
n_losses = 5    # graphite
k_losses = 4
d_list = [0.2, 0.1, 0.02, 0.1, 0.2]
n_list = [nb, n_up, n_losses + 1j*k_losses, n_down, nb]
print(n_list)
print(d_list)
lambda_list = np.linspace(2,5,100)
losses_total = np.sum(d_list * np.imag(n_list))
print(losses_total)
k_avg = losses_total / np.sum(d_list)
print(k_avg)
print(np.sum(d_list))


d_list.append(inf)
d_list.insert(0, inf)
n_list.append(1)
n_list.insert(0, 1)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

delta_lamb = lambda_list[-1] - lambda_list[0]
T_list_LR, R_list_LR, A_list_LR = TRA_func(n_list, d_list, lambda_list)
T_list_RL, R_list_RL, A_list_RL = TRA_func(n_list_reversed, d_list_reversed, lambda_list)
d_bulk = [np.sum(d_list[1:-1])]
n_bulk = [nb + 1j*k_avg]

d_bulk.append(inf)
d_bulk.insert(0, inf)
n_bulk.append(1)
n_bulk.insert(0, 1)
print(d_bulk)
print(n_bulk)

trans_bulk, refl_bulk, emiss_bulk = TRA_func(n_bulk, d_bulk, lambda_list)
print(trans_bulk)

FOM = np.trapz(T_list_RL, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list) * np.trapz(A_list_LR, x=lambda_list) / delta_lamb
ASYM = np.trapz(A_list_LR, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list)
T_avg = np.trapz(T_list_RL, x=lambda_list) / delta_lamb

# plotting using Will's plot modules

savename = f'TRA_nb={nb}_A={A}_x0={gam}_squared_off_gam_n~8_gam_k~5_a_k~0.1_shifted_n'
xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
title = "TRA for KK MS and Equivalently Lossy Bulk"
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

#%% ######################################################################################
# single-wavelength TMM calculation vs. width of square n

gam = 0.01
A = 10
nb = 2.3
gam_prop_n_arr = np.arange(1, 100, 0.1)
T_list_LR = np.zeros_like(gam_prop_n_arr)
R_list_LR = np.zeros_like(gam_prop_n_arr)
A_list_LR = np.zeros_like(gam_prop_n_arr)
T_list_RL = np.zeros_like(gam_prop_n_arr)
R_list_RL = np.zeros_like(gam_prop_n_arr)
A_list_RL = np.zeros_like(gam_prop_n_arr)
trans_bulk = np.zeros_like(gam_prop_n_arr)
emiss_bulk = np.zeros_like(gam_prop_n_arr)
lamb = 2.5
for i, gam_prop_n in enumerate(gam_prop_n_arr):
    n_list, d_list = generate_square_n_and_d(gam, A, nb, gam_prop_n=gam_prop_n, gam_prop_k=1, a_prop_k=1, plot_flag=False, save_flag=False)

    losses_total = np.sum(d_list * np.imag(n_list))
    print(i)
    trans_bulk[i] = np.exp(-4*np.pi*losses_total/lamb)
    emiss_bulk[i] = 1 - trans_bulk[i]

    d_list.append(inf)
    d_list.insert(0, inf)
    n_list.append(nb)
    n_list.insert(0, nb)

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]

    T_list_LR[i], R_list_LR[i], A_list_LR[i] = TRA_func_single(n_list, d_list, lamb)
    T_list_RL[i], R_list_RL[i], A_list_RL[i] = TRA_func_single(n_list_reversed, d_list_reversed, lamb)

ASYM_LR = A_list_LR / A_list_RL
ASYM_bulk = emiss_bulk / A_list_RL
FOM_ASYM = T_list_RL * ASYM_LR
FOM_KK = T_list_RL / A_list_RL
FOM_bulk = trans_bulk / emiss_bulk

xlabel = 't$_n$ / x$_0$'; ylabel = 'Fraction of Power'
title = "KK MS and Equiv. Lossy Bulk at $\lambda$=2.5$\mu$m"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,gam_prop_n_arr,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
plot(fig,ax,gam_prop_n_arr,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,gam_prop_n_arr,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,gam_prop_n_arr,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,gam_prop_n_arr,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,gam_prop_n_arr,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,gam_prop_n_arr,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

legend(fig,ax,auto_scale=True)

xlabel = 't$_n$ / x$_0$'; ylabel = 'Fraction of Power'
title = "KK MS and Equiv. Lossy Bulk at $\lambda$=2.5$\mu$m"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,gam_prop_n_arr,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
plot(fig,ax,gam_prop_n_arr,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

legend(fig,ax,auto_scale=True)

xlabel = 't$_n$ / x$_0$'; ylabel = 'Fraction of Power'
title = "KK MS and Equiv. Lossy Bulk at $\lambda$=2.5$\mu$m"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,gam_prop_n_arr,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,gam_prop_n_arr,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,gam_prop_n_arr,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,gam_prop_n_arr,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,gam_prop_n_arr,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

legend(fig,ax,auto_scale=True)

xlabel = 't$_n$ / x$_0$'; ylabel = ''
title = "Transmittance Times Asymmetry FOM"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,gam_prop_n_arr,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
plot(fig,ax,gam_prop_n_arr,ASYM_LR,label=r'ASYM',color=colors.red,auto_scale=True)
plot(fig,ax,gam_prop_n_arr,FOM_ASYM,label=r'FOM',color=colors.purple,auto_scale=True)

legend(fig,ax,auto_scale=True)

xlabel = 't$_n$ / x$_0$'; ylabel = 'FOM'
title = "Transmittance Over Emittance FOM"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,gam_prop_n_arr,FOM_KK,label=r'FOM$_{KK}$',color=colors.blue,auto_scale=True)
plot(fig,ax,gam_prop_n_arr,FOM_bulk,label=r'FOM$_{{bulk}}$',color=colors.red,auto_scale=True)
#plot(fig,ax,gam_prop_n_arr,R_list_RL,label=r'R$_{RL}$',color=colors.purple,auto_scale=True)

legend(fig,ax,auto_scale=True)

xlabel = 't$_n$ / x$_0$'; ylabel = 'FOM Enhancement'
title = r"FOM$_{KK}$ / FOM$_{{bulk}}$"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,gam_prop_n_arr,FOM_KK/FOM_bulk,color=colors.blue,auto_scale=True)

#%% ######################################################################################
# single-wavelength TMM calculation vs. height of n-kink

gam = 0.01
A = 10
nb = 2.3
n_kink_arr = np.arange(0.01, 1, 0.01)
T_list_LR = np.zeros_like(n_kink_arr)
R_list_LR = np.zeros_like(n_kink_arr)
A_list_LR = np.zeros_like(n_kink_arr)
T_list_RL = np.zeros_like(n_kink_arr)
R_list_RL = np.zeros_like(n_kink_arr)
A_list_RL = np.zeros_like(n_kink_arr)
trans_bulk = np.zeros_like(n_kink_arr)
emiss_bulk = np.zeros_like(n_kink_arr)
lamb = 2.5
for i, n_kink in enumerate(n_kink_arr):
    n_list, d_list = generate_square_n_and_d(gam, A, nb, gam_prop_n=8, gam_prop_k=1, a_prop_k=1, n_kink=n_kink, plot_flag=False, save_flag=False)

    losses_total = np.sum(d_list * np.imag(n_list))
    print(i)
    trans_bulk[i] = np.exp(-4*np.pi*losses_total/lamb)
    emiss_bulk[i] = 1 - trans_bulk[i]

    d_list.append(inf)
    d_list.insert(0, inf)
    n_list.append(nb)
    n_list.insert(0, nb)

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]

    T_list_LR[i], R_list_LR[i], A_list_LR[i] = TRA_func_single(n_list, d_list, lamb)
    T_list_RL[i], R_list_RL[i], A_list_RL[i] = TRA_func_single(n_list_reversed, d_list_reversed, lamb)

ASYM_LR = A_list_LR / A_list_RL
ASYM_bulk = emiss_bulk / A_list_RL
FOM_ASYM = T_list_RL * ASYM_LR
FOM_KK = T_list_RL / A_list_RL
FOM_bulk = trans_bulk / emiss_bulk

xlabel = '$\Delta$n'; ylabel = 'Fraction of Power'
title = "KK MS and Equiv. Lossy Bulk at $\lambda$=2.5$\mu$m"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,n_kink_arr,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
plot(fig,ax,n_kink_arr,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,n_kink_arr,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,n_kink_arr,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,n_kink_arr,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,n_kink_arr,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,n_kink_arr,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

legend(fig,ax,auto_scale=True)

xlabel = '$\Delta$n'; ylabel = 'Fraction of Power'
title = "KK MS and Equiv. Lossy Bulk at $\lambda$=2.5$\mu$m"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,n_kink_arr,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
plot(fig,ax,n_kink_arr,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

legend(fig,ax,auto_scale=True)


xlabel = '$\Delta$n'; ylabel = 'Fraction of Power'
title = "KK MS and Equiv. Lossy Bulk at $\lambda$=2.5$\mu$m"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,n_kink_arr,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,n_kink_arr,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,n_kink_arr,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,n_kink_arr,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,n_kink_arr,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

legend(fig,ax,auto_scale=True)


xlabel = '$\Delta$n'; ylabel = ''
title = "Transmittance Times Asymmetry FOM"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,n_kink_arr,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
plot(fig,ax,n_kink_arr,ASYM_LR,label=r'ASYM',color=colors.red,auto_scale=True)
plot(fig,ax,n_kink_arr,FOM_ASYM,label=r'FOM',color=colors.purple,auto_scale=True)

legend(fig,ax,auto_scale=True)

xlabel = '$\Delta$n'; ylabel = 'FOM'
title = "Transmittance Over Emittance FOM"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,n_kink_arr,FOM_KK,label=r'FOM$_{KK}$',color=colors.blue,auto_scale=True)
plot(fig,ax,n_kink_arr,FOM_bulk,label=r'FOM$_{{bulk}}$',color=colors.red,auto_scale=True)
#plot(fig,ax,gam_prop_n_arr,R_list_RL,label=r'R$_{RL}$',color=colors.purple,auto_scale=True)

legend(fig,ax,auto_scale=True)

xlabel = '$\Delta$n'; ylabel = 'FOM Enhancement'
title = r"FOM$_{KK}$ / FOM$_{{bulk}}$"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,n_kink_arr,FOM_KK/FOM_bulk,color=colors.blue,auto_scale=True)
#%% ############################################################################################

# sweep over minimum thickness of layer to observe changes in FOM and zero-reflectance

gam = 0.1
A = 2
nb = 2.7
min_thickness_arr = np.logspace(-3, -1, num=100)
FOM_arr = np.zeros_like(min_thickness_arr)
ASYM_arr = np.zeros_like(min_thickness_arr)
T_avg_arr = np.zeros_like(min_thickness_arr)
refl_arr = np.zeros_like(min_thickness_arr)
k_avg_arr = np.zeros_like(min_thickness_arr)
lambda_list = np.linspace(2,5,100)
delta_lamb = lambda_list[-1] - lambda_list[0]

for i, min_thickness in enumerate(min_thickness_arr):
    print(f"minimum thickness is: {min_thickness}")
    n_list, d_list, num_layers, k_avg_arr[i] = generate_n_and_d(gam, A, nb, False, min_thickness=min_thickness)

    d_list.append(inf)
    d_list.insert(0, inf)
    n_list.append(nb)
    n_list.insert(0, nb)

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]

    T_list_LR, R_list_LR, A_list_LR = TRA_func(n_list, d_list, lambda_list)
    T_list_RL, R_list_RL, A_list_RL = TRA_func(n_list_reversed, d_list_reversed, lambda_list)

    FOM_arr[i] = np.trapz(T_list_RL, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list) * np.trapz(A_list_LR, x=lambda_list) / delta_lamb
    ASYM_arr[i] = np.trapz(A_list_LR, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list)
    T_avg_arr[i] = np.trapz(T_list_RL, x=lambda_list) / delta_lamb
    refl_arr[i] = np.trapz(R_list_LR, x=lambda_list) / delta_lamb

#%% ############################################################################################
# plot above results

savename = f'FOM_Tavg_ASYM_vs_min_thickness0.001~0.1_A={A}_x0={gam}'
xlabel = 'Minimum Thickness ($\mu$m)'; ylabel = ''
title = f'Effects of Minimum Layer Thickness'
fig,ax = plot_setup(xlabel,ylabel,title=title, xscale='log', figsize=(5,4),auto_scale=True)

plot(fig,ax,min_thickness_arr,T_avg_arr,label=r'T$_{avg}$',color=colors.blue,auto_scale=True)
plot(fig,ax,min_thickness_arr,ASYM_arr,label=r'ASYM',color=colors.red,auto_scale=True)
plot(fig,ax,min_thickness_arr,FOM_arr,label=r'FOM',color=colors.green,auto_scale=True)

plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
legend(fig,ax,auto_scale=True)
plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

savename = f'k_avg_vs_min_thickness0.001~0.1_A={A}_x0={gam}'
xlabel = 'Minimum Thickness ($\mu$m)'; ylabel = 'Average k'
title = f'Average k vs. Minimum Layer Thickness'
fig,ax = plot_setup(xlabel,ylabel,title=title, xscale='log', figsize=(5,4),auto_scale=True)
plot(fig,ax,min_thickness_arr,k_avg_arr,color=colors.blue,auto_scale=True)
plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)

savename = f'Refl_avg_vs_min_thickness0.001~0.1_A={A}_x0={gam}'
xlabel = 'Minimum Thickness ($\mu$m)'; ylabel = 'Average Reflectance'
title = f'Average R$_{{LR}}$ vs. Minimum Thickness'
fig,ax = plot_setup(xlabel,ylabel,title=title, xscale='log', yscale= 'log', figsize=(5,4),auto_scale=True)
plot(fig,ax,min_thickness_arr,refl_arr,color=colors.blue,auto_scale=True)
plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)

#%% ############################################################################################

# sweep over amplitude A, and plot asymmetry as a function of transmittance

gam = 0.1
A_arr = np.arange(1, 15, 0.1)
nb = 2.7
min_thickness = 0.01
FOM_arr = np.zeros_like(A_arr)
FOM_old_arr = np.zeros_like(A_arr)
FOM_bulk_arr = np.zeros_like(A_arr)
ASYM_arr = np.zeros_like(A_arr)
T_avg_arr = np.zeros_like(A_arr)
k_avg_arr = np.zeros_like(A_arr)
lambda_list = np.linspace(2,5,100)
delta_lamb = lambda_list[-1] - lambda_list[0]

for i, A in enumerate(A_arr):
    print(f"A is: {A}")
    n_list, d_list, num_layers, k_avg_arr[i] = generate_n_and_d(gam, A, nb, False, min_thickness=min_thickness)

    d_list.append(inf)
    d_list.insert(0, inf)
    n_list.append(nb)
    n_list.insert(0, nb)

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]

    T_list_LR, R_list_LR, A_list_LR = TRA_func(n_list, d_list, lambda_list)
    T_list_RL, R_list_RL, A_list_RL = TRA_func(n_list_reversed, d_list_reversed, lambda_list)

    FOM_arr[i] = np.trapz(T_list_RL, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list) * np.trapz(A_list_LR, x=lambda_list) / delta_lamb
    FOM_old_arr[i] = (np.trapz(T_list_RL, x=lambda_list))**2 / np.trapz(A_list_RL, x=lambda_list) / delta_lamb
    FOM_bulk_arr[i] = (np.trapz(T_list_RL, x=lambda_list))**2 / np.trapz(A_list_LR, x=lambda_list) / delta_lamb
    ASYM_arr[i] = np.trapz(A_list_LR, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list)
    T_avg_arr[i] = np.trapz(T_list_RL, x=lambda_list) / delta_lamb

#%% ############################################################################################

# plot above results

savename = f'ASYM_vs_Tavg_A=1~15_x0={gam}_min_thickness_{min_thickness}'
xlabel = 'Transmittance'; ylabel = 'Emission Asymmetry'
title = f'Emission Asymmetry vs. Transmittance'
fig,ax = plot_setup(xlabel,ylabel,title=title, figsize=(5,4),auto_scale=True)

plot(fig,ax,T_avg_arr,ASYM_arr,color=colors.blue,auto_scale=True)

plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)

savename = f'FOM_vs_A=1~15_x0={gam}_min_thickness_{min_thickness}'
xlabel = 'A'; ylabel = 'FOM'
title = f'FOM vs. Lorentzian Amplitude A'
fig,ax = plot_setup(xlabel,ylabel,title=title, figsize=(5,4),auto_scale=True)

plot(fig,ax,A_arr,FOM_arr,color=colors.blue,auto_scale=True)

plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)

savename = f'FOM_old_vs_Tavg_A=1~15_x0={gam}_min_thickness_{min_thickness}'
xlabel = 'Transmittance'; ylabel = 'Old FOM (T$^2$/$\epsilon$)'
title = f'Old FOM vs. Transmittance'
fig,ax = plot_setup(xlabel,ylabel,title=title, yscale='log',figsize=(5,4),auto_scale=True)

plot(fig,ax,T_avg_arr,FOM_old_arr, label='FOM$_{{KK}}$', color=colors.blue,auto_scale=True)
plot(fig,ax,T_avg_arr,FOM_bulk_arr, '--', label='FOM$_{{bulk}}$', color=colors.light_blue,auto_scale=True)

plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
legend(fig,ax,auto_scale=True)
plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

savename = f'FOM_enhancement_vs_Tavg_A=1~15_x0={gam}_min_thickness_{min_thickness}'
xlabel = 'Transmittance'; ylabel = 'FOM enhancement'
title = f'FOM Enhancement vs. Transmittance'
fig,ax = plot_setup(xlabel,ylabel,title=title, figsize=(5,4),auto_scale=True)

plot(fig,ax,T_avg_arr,FOM_old_arr / FOM_bulk_arr,color=colors.blue,auto_scale=True)

plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)

#%% ###########################################################################

n_sapphire_data = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\RI\\alumina_n_1~14um_Kischkat.txt', skiprows=1)
k_sapphire_data = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\RI\\alumina_k_1~14um_Kischkat.txt', skiprows=1)

wls = np.linspace(2,5,100)
n_sapphire_interp = interp1d(n_sapphire_data[:,0], n_sapphire_data[:,1], kind='linear', fill_value='extrapolate')
k_sapphire_interp = interp1d(k_sapphire_data[:,0], k_sapphire_data[:,1], kind='linear', fill_value='extrapolate')

n_sapphire = n_sapphire_interp(wls)
k_sapphire = k_sapphire_interp(wls)

#%% ################################################################################

xlabel = 'Wavelength ($\mu$m)'; ylabel = 're(n)'
title = "Sapphire n data"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,n_sapphire_data[:,0],n_sapphire_data[:,1],color=colors.blue,auto_scale=True)
plot(fig,ax,wls,n_sapphire, '--', color=colors.red,auto_scale=True)

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'im(n)'
title = "Sapphire k data"
fig,ax = plot_setup(xlabel,ylabel,xlim=(2,5),title=title,figsize=(5,4),auto_scale=True)

#plot(fig,ax,n_ZnS_data[:,0],n_ZnS_data[:,1],color=colors.blue,auto_scale=True)
plot(fig,ax,wls,k_sapphire, '--', color=colors.red,auto_scale=True)

#%% ###################################################################################

d_list = [np.inf, 200, np.inf]
c_list = ['i', 'i', 'i']
T_list = np.zeros_like(wls)
R_list = np.zeros_like(wls)
A_list = np.zeros_like(wls)

for i, lamb in enumerate(wls):

    n_list = [1, n_sapphire[i] + 1j*k_sapphire[i], 1]

    T_list[i] = tmm.tmm_core.inc_tmm('s', n_list, d_list, c_list, 0, lamb)['T']
    R_list[i] = tmm.tmm_core.inc_tmm('s', n_list, d_list, c_list, 0, lamb)['R']
    A_list[i] = 1 - T_list[i] - R_list[i]


xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
title = "TRA of Window (no coating)"
fig,ax = plot_setup(xlabel,ylabel,xlim=(2,5),title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,wls,T_list, color=colors.blue, label='T', auto_scale=True)
plot(fig,ax,wls,R_list, color=colors.green, label='R', auto_scale=True)
plot(fig,ax,wls,A_list, color=colors.red, label='A', auto_scale=True)

legend(fig,ax,auto_scale=True)

#%% ######################################################################################
# wavelength-sweep single TMM calculation for TRA, including plotting

gam = 0.01
A = 1
nb = 2.3
min_thickness=0.001
m = 1e-5
b = 1.5e-6
pol = 'p'
angle = 0
lambda_list = np.linspace(2,5,100)
T_list_LR = np.zeros_like(lambda_list)
R_list_LR = np.zeros_like(lambda_list)
A_list_LR = np.zeros_like(lambda_list)
T_list_RL = np.zeros_like(lambda_list)
R_list_RL = np.zeros_like(lambda_list)
A_list_RL = np.zeros_like(lambda_list)
trans_bulk = np.zeros_like(lambda_list)
refl_bulk = np.zeros_like(lambda_list)
emiss_bulk = np.zeros_like(lambda_list)
k_val = np.zeros_like(lambda_list)
#n_list, d_list, num_layers = generate_n_and_d(gam, a, nb, min_thickness=min_thickness, k_threshold_ratio=0.5, save_flag=False)

#print(n_list)
#print(d_list)
#print(c_list)
n_list, d_list, count = generate_n_and_d(gam, A, nb, min_thickness=min_thickness, plot_flag=True)
for i, lamb in enumerate(lambda_list):
    n_list, d_list, count = generate_n_and_d(gam, A, nb, min_thickness=min_thickness, plot_flag=False)
    losses_total = round(np.sum(d_list * np.imag(n_list)), 3)
    c_list = ['c']
    for j in range(count):
        c_list.append('c')
    #n_list = []
    #d_list = []
    #c_list = []
    k_val[i] = m*(lamb - 2) + b
    # add bulk window before spatial KK coating
    d_list.insert(0, 2000)
    n_list.insert(0, nb + k_val[i]*1j)
    c_list.insert(0, 'i')

    # add semi-infinite air layers
    d_list.append(inf)
    d_list.insert(0, inf)
    n_list.append(1)            # change these to nb to see broadband zero-reflectance
    n_list.insert(0, 1)
    c_list.append('i')
    c_list.insert(0, 'i')

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]
    c_list_reversed = c_list[::-1]

    d_bulk = [np.inf, 2000, np.inf]
    c_bulk = ['i', 'i', 'i']
    n_bulk = [1, nb + k_val[i]*1j, 1]

    delta_lamb = lambda_list[-1] - lambda_list[0]
    T_list_LR[i] = tmm.tmm_core.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)['T']
    R_list_LR[i] = tmm.tmm_core.inc_tmm(pol, n_list, d_list, c_list, angle, lamb)['R']
    A_list_LR[i] = 1 - T_list_LR[i] - R_list_LR[i]

    T_list_RL[i] = tmm.inc_tmm(pol, n_list_reversed, d_list_reversed, c_list_reversed, angle, lamb)['T']
    R_list_RL[i] = tmm.inc_tmm(pol, n_list_reversed, d_list_reversed, c_list_reversed, angle, lamb)['R']
    A_list_RL[i] = 1 - T_list_RL[i] - R_list_RL[i]

    trans_bulk[i] = tmm.tmm_core.inc_tmm(pol, n_bulk, d_bulk, c_bulk, angle, lamb)['T']
    refl_bulk[i] = tmm.tmm_core.inc_tmm(pol, n_bulk, d_bulk, c_bulk, angle, lamb)['R']
    emiss_bulk[i] = 1 - trans_bulk[i] - refl_bulk[i]

FOM = np.trapz(T_list_RL, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list) * np.trapz(A_list_LR, x=lambda_list) / delta_lamb
ASYM = np.trapz(A_list_LR, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list)
T_avg = np.trapz(T_list_RL, x=lambda_list) / delta_lamb

# plotting using Will's plot modules

savename = f'TRA_nb={nb}_A={A}_x0={gam}_squared_off_gam_n~8_gam_k~5_a_k~0.1_shifted_n'
xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
#title = f'TRA (A={A}, x$_0$={gam}$\mu$m)'
#title = "Window without Coating"
title = f"Window with Spatial KK Coating (losses = {losses_total})"
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,T_list_LR,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,lambda_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,refl_bulk, '--', label=r'R$_{bulk}$',color=colors.light_green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$ / A$_{{RL}}$ = {round(ASYM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

#plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
legend(fig,ax,auto_scale=True)

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'im(n)'
title = "Window's Losses"
fig,ax = plot_setup(xlabel,ylabel,xlim=(2,5),title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,k_val, '--', color=colors.red, label='TMM', auto_scale=True)

# %%
