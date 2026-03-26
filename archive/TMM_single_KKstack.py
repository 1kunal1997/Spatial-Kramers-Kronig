#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tmm

from numpy import pi, linspace, inf, array
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.signal import hilbert
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Load external plotting functions
from plot_functions import plot_setup, plot, legend

# Load plotting colors
import colors # make available colors from schmid_colors.py 

# Image file settings
fmt = '.png' # image format (use png for PowerPoint, pdf and eps for publications)
dpi = 300 # image resolution, density of pixels per inch (use at least 300)
fig_dir = 'C:\\Users\\kl89\\MS Window Project\\Figures\\'

degree = pi/180


#%% ##########################################################################################

# generate nk for spatial KK stack

nb = 1.7

def eps(x, a, gam, nb):
    return nb**2 - a * gam / (x + 1j*gam)


#%% #############################################################################################
# function to generate list of refractive indices and thicknesses of each layer
# in TMM calculation

def generate_n_and_d(gam, a, nb, plot_flag, n_prop, x_prop):    
    
    dx      = gam/100               # Step size in 'continuous' Lorentzian
    xmin    = -gam * 200           # Limits of Lorentzian
    xmax    = - xmin

    nx      = 1 + int(np.floor((xmax - xmin) / dx))
    xx      = np.linspace(xmin, xmax, nx)
    ee      = eps(xx,a,gam,nb)                    # Smooth Lorentzian curve
    nk      = np.sqrt(ee)

    k_max   = np.max(np.imag(nk))     # Max k value. used to set max n-step size 
    del_n   = n_prop*k_max/15                # Max n-step size in discrete Lorentzian approximation
    del_x   = x_prop*25*gam                # Max x-step size in discrete Lorentzian approximation

    xq      = [xx[0]]                               
    nq      = [nk[0]]
    count   = 0
    for k in range(0,nx):
        if abs((nk[k]) - (nq[count])) > del_n or abs((xx[k]) - (xq[count])) > del_x:
            xq.append(xx[k])
            nq.append(nk[k])
            count = count + 1

    xq = np.append(xq,xmax)     # should we be appending xx[-1]? because xx does not include xmax as it is rn
    nq = np.append(nq,nk[-1])

    d_list = np.diff(xq)
    n_list = (nq[:-1] + nq[1:]) / 2
    k_avg = np.sum(d_list * np.imag(n_list)) / np.sum(d_list)
    
    # plot imaginary and real part of refractive index
    if (plot_flag):

        savename = f'Ref_index_real_A~{a}_gam~{gam}'
        xlabel = 'x/x$_0$'; ylabel = 'Refractive Index'
        title = f'Real Refractive Index (A={a}, x$_0$={gam})'
        fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True, xlim=(xmin/gam,xmax/gam))

        n_real = np.real(n_list)
        midpoints = (xq[:-1] + xq[1:]) / 2
        plot(fig,ax, xx/gam, np.real(nk), label='smooth', color=colors.red,auto_scale=True)
        ax.stairs(n_real, xq/gam, baseline=nb, label='discrete', linewidth = 2)
        plot(fig,ax, midpoints/gam, n_real, '*', markersize=7, label='inputs', color=colors.green,auto_scale=True)

        plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)

        savename = f'Ref_index_imag_A~{a}_gam~{gam}'
        title = f'Imaginary Refractive Index (A={a}, x$_0$={gam})'
        fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True, xlim=(xmin/gam,xmax/gam),ylim=(-k_max/20,k_max*(1+1/20)))

        n_imag = np.imag(n_list)
        plot(fig,ax, xx/gam, np.imag(nk), label='smooth', color=colors.red,auto_scale=True)
        ax.stairs(n_imag, xq/gam, baseline=0, label='discrete', linewidth = 2)
        plot(fig,ax, midpoints/gam, n_imag, '*', markersize=7, label='inputs', color=colors.green,auto_scale=True)

        plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
        legend(fig,ax,auto_scale=True) # create legend from curves labeled above
        plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

    return (n_list.tolist(), d_list.tolist(), count, k_avg)

#%% #######################################################################################
# sample run of generate_n_and_d()

gam = 0.001
A = 5
nb = 1.7

# x_prop_max = 5, min = 0.1
# n_prop_max = 20, min = 0.1
generate_n_and_d(gam, A, nb, True, 1, 1);

#%% ######################################################################################
# function to calculate TRA of given n_list and d_list

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

#%% ######################################################################################
# function to calculate TRA of given n_list and d_list

def TRA_func_angle(n_list, d_list, angle_list, lamb, pol):
    T_list = np.zeros_like(angle_list)
    R_list = np.zeros_like(angle_list)
    A_list = np.zeros_like(angle_list)
    
    for j, angle in enumerate(angle_list):
        T_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['T']
        R_list[j] = tmm.coh_tmm(pol, n_list, d_list, angle, lamb)['R']
        A_list[j] = 1 - T_list[j] - R_list[j]

    return (T_list, R_list, A_list)

#%% ######################################################################################
# single TMM calculation for TRA, including plotting

gam = 0.01
A = 15
nb = 2.7
lambda_list = np.linspace(2,5,100)
n_list, d_list, count, k_avg = generate_n_and_d(gam, A, nb, True, 1, 1)
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


T_list_LR, R_list_LR, A_list_LR = TRA_func(n_list, d_list, lambda_list)
T_list_RL, R_list_RL, A_list_RL = TRA_func(n_list_reversed, d_list_reversed, lambda_list)

savename = f'TRA_A={A}_x0={gam}'
xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
title = f''
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,lambda_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

#plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
legend(fig,ax,auto_scale=True)
#plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

#%% ######################################################################################
# single TMM calculation for TRA, including plotting

gam = 0.01
A = 15
nb = 2.7
pol = 's'
lamb = 3
angle_list = np.arange(0, 80, 1)

n_list, d_list, count, k_avg = generate_n_and_d(gam, A, nb, False, 1, 1)
losses_total = np.sum(d_list * np.imag(n_list))
print(losses_total)
trans_bulk = np.exp(-4*np.pi*losses_total/lamb/np.cos(angle_list*degree))
emiss_bulk = 1 - trans_bulk

d_list.append(inf)
d_list.insert(0, inf)
n_list.append(nb)
n_list.insert(0, nb)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

T_list_LR, R_list_LR, A_list_LR = TRA_func_angle(n_list, d_list, angle_list*degree, lamb, pol=pol)
T_list_RL, R_list_RL, A_list_RL = TRA_func_angle(n_list_reversed, d_list_reversed, angle_list*degree, lamb, pol=pol)

savename = f'p - polarization'
xlabel = 'Angle of Incidence'; ylabel = 'Fraction of Power'
title = f's - polarization'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(angle_list[0],angle_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,angle_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
plot(fig,ax,angle_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,angle_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,angle_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
plot(fig,ax,angle_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,angle_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,angle_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

#plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
legend(fig,ax,auto_scale=True)
#plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

#%% #####################################################################################

gam = 0.01
A = 5
nb = 1.7
convergence_threshold = 1e-3
lambda_list = np.linspace(2,5,100)

#n_prop_arr = [0.2, 0.4, 0.6, 0.8, 1]
#x_prop_arr = np.arange(0.1, 2.05, step=0.05)
n_prop_arr = np.arange(0.1, 10.05, step=0.05)
x_prop_arr = [0.1, 0.5, 1, 1.5, 2]

mre_T_arr = np.zeros((len(x_prop_arr), len(n_prop_arr)), dtype=float)
mre_A_arr = np.zeros((len(x_prop_arr), len(n_prop_arr)), dtype=float)
num_points_diff = np.zeros((len(x_prop_arr), len(n_prop_arr)), dtype=int)
num_points = np.zeros((len(x_prop_arr), len(n_prop_arr)), dtype=int)

for i, x_prop in enumerate(x_prop_arr):
    print(f'x_prop is: {x_prop}')

    n_list, d_list, count, k_avg = generate_n_and_d(gam, A, nb, False, n_prop_arr[0], x_prop)

    d_list.append(inf)
    d_list.insert(0, inf)
    n_list.append(nb)
    n_list.insert(0, nb)

    n_list_reversed = n_list[::-1]
    d_list_reversed = d_list[::-1]

    T_list_LR, R_list_LR, A_list_LR = TRA_func(n_list, d_list, lambda_list)
    T_list_RL, R_list_RL, A_list_RL = TRA_func(n_list_reversed, d_list_reversed, lambda_list)

    for j, n_prop in enumerate(n_prop_arr):
        print(f'n_prop is: {n_prop}')

        n_list, d_list, count_new, k_avg = generate_n_and_d(gam, A, nb, False, n_prop, x_prop)

        d_list.append(inf)
        d_list.insert(0, inf)
        n_list.append(nb)
        n_list.insert(0, nb)

        n_list_reversed = n_list[::-1]
        d_list_reversed = d_list[::-1]

        T_list_LR_new, R_list_LR_new, A_list_LR_new = TRA_func(n_list, d_list, lambda_list)
        T_list_RL_new, R_list_RL_new, A_list_RL_new = TRA_func(n_list_reversed, d_list_reversed, lambda_list)

        def mean_relative_error(lst, lst_new):
            relative_error = np.abs(lst_new - lst) / np.abs(lst)
            mean_relative_error = np.mean(relative_error)
            return mean_relative_error

        # Compute Mean Relative Error
        mre_T_arr[i][j] = mean_relative_error(T_list_RL, T_list_RL_new)
        mre_A_arr[i][j] = mean_relative_error(A_list_RL, A_list_RL_new)
        num_points_diff[i][j] = count - count_new
        num_points[i][j] = count_new
        count = count_new

        T_list_LR, R_list_LR, A_list_LR = T_list_LR_new, R_list_LR_new, A_list_LR_new
        T_list_RL, R_list_RL, A_list_RL = T_list_RL_new, R_list_RL_new, A_list_RL_new

#%% ###########################################################################################

np.savetxt("C:\\Users\\kl89\\MS Window Project\\Data\\MRE_Trans_n_prop0.1~10.0_hold_x_prop0.1~2.txt", mre_T_arr)
np.savetxt("C:\\Users\\kl89\\MS Window Project\\Data\\MRE_Abs_n_prop0.1~10.0_hold_x_prop0.1~2.txt", mre_A_arr)
np.savetxt("C:\\Users\\kl89\\MS Window Project\\Data\\Num_points_n_prop0.1~10.0_hold_x_prop0.1~2.txt", num_points)
np.savetxt("C:\\Users\\kl89\\MS Window Project\\Data\\Num_points_diff_n_prop0.1~10.0_hold_x_prop0.1~2.txt", num_points_diff)

#%% #############################################################################################
# plot MRE of T and A as a colorplot
extent=(0.001, 0.1, 0, 50)
plt.figure()
plt.imshow(np.log((mre_T_arr).T), interpolation='none', aspect='auto', origin='lower')
plt.ylabel('Amplitude A')
plt.xlabel('Width x$_0$ ($\mu$m)')
plt.title('FoM Enhancement for Lorentzian $\epsilon^{\'\'}$')
ax = plt.gca()
plt.colorbar()

#%% ###########################################################################################
# plot MRE as a 1D plot

color_arr = [colors.blue, colors.red, colors.green, colors.purple, colors.copper]
savename = f'MRE_and_num_layers_Abs_n_prop0.5~2.0_hold_x_prop1,1.5'
xlabel = 'n$_{{prop}}$'; ylabel = 'MRE'; y2label = 'Number of Layers'
title = f'Mean Relative Error of Abs. from n$_{{prop}}$'
fig,ax,ax2 = plot_setup(xlabel,ylabel,twin_axis=True, y2label=y2label,title=title,figsize=(5,4),auto_scale=True)

for i, x_prop in enumerate(x_prop_arr[1:3]):
    plot(fig,ax, n_prop_arr[10:40], mre_A_arr[i+1][10:40], label=f'x$_{{prop}}$={x_prop}',color=color_arr[i],auto_scale=True)
    plot(fig,ax2, n_prop_arr[10:40], num_points[i+1][10:40], '--', label=f'x$_{{prop}}$={x_prop}',color=color_arr[i],auto_scale=True)
plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
legend(fig,ax,auto_scale=True)
plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

#%% ###########################################################################################
# plot MRE as a 1D plot

color_arr = [colors.blue, colors.red, colors.green, colors.purple, colors.copper]
savename = f'MRE_vs_num_layers_Abs_n_prop0.1~2.0_hold_x_prop0.1~2.0'
xlabel = 'Number of Layers'; ylabel = 'MRE'
title = f'Mean Relative Error of Abs. from n$_{{prop}}$'
#max_layers = np.max(num_points[:][10:60])
#print(max_layers)
#min_layers = np.min(num_points[:][10:60])
#fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),xmin=min_layers-6, xmax=max_layers, xstep=50, auto_scale=True)
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4), auto_scale=True)

for i, x_prop in enumerate(x_prop_arr):
    plot(fig,ax, num_points[i][60:], mre_A_arr[i][60:], label=f'x$_{{prop}}$={x_prop}',color=color_arr[i],auto_scale=True)

plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
legend(fig,ax,auto_scale=True)
plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

#%% ############################################################################################

nb = 1.7
gam_list = np.linspace(0.001, 0.1, num=100)
A_list = np.linspace(0, 50, num=100)
length_A = len(A_list)
FOM_bulk = np.zeros((len(gam_list), len(A_list)))
FOM_KK = np.zeros((len(gam_list), len(A_list)))

start = time.time()
for i, gam in enumerate(gam_list):
    for j, A in enumerate(A_list):
        d_list, n_list = generate_n_and_d(gam, A, nb, False)
        (T_list_LR, R_list_LR, A_list_LR, T_list_RL, R_list_RL, A_list_RL) = TRA_func(n_list, d_list)
        lambda_list = np.linspace(2,5,100)
        delta_lamb = lambda_list[-1] - lambda_list[0]
        
        losses_total = np.trapz(np.imag(n_list), x=d_list)
        trans_bulk = np.exp(-4*np.pi*losses_total/lambda_list)
        emiss_bulk = 1 - trans_bulk
        FOM_bulk[i][j] = (np.trapz(trans_bulk, x=lambda_list))**2 / np.trapz(emiss_bulk, x=lambda_list) / delta_lamb
        
        FOM_LR = (np.trapz(T_list_LR, x=lambda_list))**2 / np.trapz(A_list_LR, x=lambda_list) / delta_lamb
        FOM_RL = (np.trapz(T_list_RL, x=lambda_list))**2 / np.trapz(A_list_RL, x=lambda_list) / delta_lamb
        FOM_KK[i][j] = max(FOM_LR, FOM_RL)
        print(i*length_A + j)

end = time.time()
print(f"Duration: {end - start} seconds")
np.savetxt("C:\\Users\\kl89\\MS Window Project\\FOM_KK_singleLorentz_vs_Aandgam.txt", FOM_KK)


# In[5]:


def generate_x_from_d(d_list):
    x_list = np.zeros(len(d_list)+1)
    curr = 0
    for i, thickness in enumerate(d_list):
        x_list[i+1] = curr + thickness
        curr = x_list[i+1]
    return x_list


# In[7]:


for i, gam in enumerate(gam_list):
    for j, A in enumerate(A_list):
        d_list, n_list = generate_n_and_d(gam, A, nb, False)
        x_list = generate_x_from_d(d_list[1:-1])
        losses_total = np.trapz(np.imag(n_list[:-1]), x=x_list)

        lambda_list = np.linspace(2,5,100)
        delta_lamb = lambda_list[-1] - lambda_list[0]
        trans_bulk = np.exp(-4*np.pi*losses_total/lambda_list)
        emiss_bulk = 1 - trans_bulk
        FOM_bulk[i][j] = (np.trapz(trans_bulk, x=lambda_list))**2 / np.trapz(emiss_bulk, x=lambda_list) / delta_lamb
        print(i*len(A_list) + j)


# In[33]:


plt.figure()
plt.imshow((FOM_KK[:,1:]/FOM_bulk[:,1:]).T, interpolation='none', aspect='auto', origin='lower', extent=(0.001, 0.1, 0, 50))
plt.ylabel('Amplitude A')
plt.xlabel('Width x$_0$ ($\mu$m)')
plt.title('FoM Enhancement for Lorentzian $\epsilon^{\'\'}$')
ax = plt.gca()
plt.colorbar()


# In[32]:


np.savetxt("C:\\Users\\kl89\\MS Window Project\\FOM_bulk_singleLorentz_vs_Aandgam.txt", FOM_bulk)
FOM_KK_read = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\FOM_KK_singleLorentz_vs_Aandgam.txt')
FOM_bulk_read = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\FOM_bulk_singleLorentz_vs_Aandgam.txt')

plt.figure()
plt.imshow((FOM_KK_read[:,1:]/FOM_bulk_read[:,1:]).T, interpolation='none', aspect='auto', origin='lower', extent=(0.001, 0.1, 0, 50))
plt.ylabel('Amplitude A')
plt.xlabel('Width x$_0$ ($\mu$m)')
plt.title('FoM Enhancement for Lorentzian $\epsilon^{\'\'}$')
ax = plt.gca()
plt.colorbar()


# In[58]:


FOM_enhancement = (FOM_KK[:,1:]/FOM_bulk[:,1:]).T
mn = np.min(FOM_enhancement)
max_index = np.argmin(FOM_enhancement)
print(np.unravel_index(max_index, FOM_enhancement.shape))
print(mn)


# In[63]:


d_list, n_list = generate_n_and_d(0.01, 10, nb, True)
x_list = generate_x_from_d(d_list[1:-1])
losses_total = np.trapz(np.imag(n_list[:-1]), x=x_list)

lambda_list = np.linspace(2,5,100)
delta_lamb = lambda_list[-1] - lambda_list[0]
trans_bulk = np.exp(-4*np.pi*losses_total/lambda_list)
emiss_bulk = 1 - trans_bulk
curr_FOM_bulk = (np.trapz(trans_bulk, x=lambda_list))**2 / np.trapz(emiss_bulk, x=lambda_list) / delta_lamb

(T_list_LR, R_list_LR, A_list_LR, T_list_RL, R_list_RL, A_list_RL) = TRA_func(n_list, d_list)

FOM_LR = (np.trapz(T_list_LR, x=lambda_list))**2 / np.trapz(A_list_LR, x=lambda_list) / delta_lamb
FOM_RL = (np.trapz(T_list_RL, x=lambda_list))**2 / np.trapz(A_list_RL, x=lambda_list) / delta_lamb
print(FOM_LR)
print(FOM_RL)
print(curr_FOM_bulk)


# In[54]:


plt.figure()
plt.plot(lambda_list, T_list_LR, label = 'T_KK')
plt.plot(lambda_list, A_list_LR, label = 'A_KK,LR')
plt.plot(lambda_list, A_list_RL, label = 'A_KK,RL')
plt.plot(lambda_list, trans_bulk, '--', label = 'T_bulk')
plt.plot(lambda_list, emiss_bulk, '--', label = 'A_bulk')
plt.legend(loc='center right')
plt.title('Transmittance and Absorbance (lorentzian KK)')
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Fraction of Power')


# In[71]:


plt.plot(lorentz_d_list, curr_FOM_vs_d/9.312)
plt.xlabel('x$_s$ ($\mu$m)')
plt.ylabel('FoM Enhancement')


# In[ ]:




