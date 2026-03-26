#%% STREED FORMULA
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import FormatStrFormatter
from os import listdir

# Directories
script_dir = os.path.dirname(__file__) 
parent_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(parent_dir, 'Data/')

# Load external plotting functions
from plot_functions import plot_setup, plot, legend, contour

# Load plotting colors
import colors # make available colors from schmid_colors.py 

# Image file settings
fmt = '.png' # image format (use png for PowerPoint, pdf and eps for publications)
dpi = 300 # image resolution, density of pixels per inch (use at least 300)
auto_scale = True
figsize=(5,3.5)

# Reset plot style to default
plt.close('all')
plt.style.use('default') # Set to default style for viewing ease in interactive window

# Load data
file_dir = 'Formula STREED'
fig_dir = os.path.join(parent_dir, 'Figures/Matplotlib/'+file_dir)
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

filename = 'STREED_formula_v1.csv'
data_full = np.genfromtxt(data_dir+file_dir+'/'+filename,delimiter=',',skip_header=5) 

# Unique values of intensities and flow rates
I_in = np.unique(data_full[:,0]) # I_in
V_feed = np.unique(data_full[:,1]) # V_feed

#%%
# Loop over intensities and flow rates
for n in range(len(I_in)):

    # Get data matrix for current intensity
    data = data_full[data_full[:,0]==I_in[n]]

    # Data columns for current intensity
    V_ratio = data[:,2] #V_air/V_feed
    SWP = data[:,3] # kg/kWh
    T_max = data[:,4] # degC
    P_in = data[:,5] # W

    # Greek letter efficiencies
    eta = data[:,7]
    xi = data[:,8]
    lam = data[:,9]
    alpha = data[:,10]
    eta_prime = data[:,11]
    xi_prime = data[:,12]
    beta = data[:,13]
    gamma = data[:,14]
    SWP = data[:,17]
    LkWh = data[:,18]

    # GOR formulas
    GOR1 = gamma*(eta_prime/(1-beta*(eta_prime+xi_prime))) # original with primes
    GOR2 = gamma*(eta_prime/(1-beta*(eta_prime+xi))) # no xi_prime (no alpha)
    GOR3 = gamma*(eta/(1-beta*(eta+xi_prime))) # no eta_prime (no lambda)
    GOR4 = gamma*(eta/(1-beta*(eta+xi))) # no xi_prime, no eta_prime

    # Plot 1
    imag_name = f'STREED_v1_formula_effs_I_in-{I_in[n]:0.0f}W'
    title = r'Heat exchanger, $I_{\rm in}$ = '+f'{I_in[n]:0.0f} W'
    xlabel = r'Flow rate ratio $V_{\rm air}/V_{\rm feed}$'
    ylabel = r'Efficiency'
    fig, ax = plot_setup(xlabel, ylabel,auto_scale=auto_scale, figsize=figsize, xlim=(100,1000),ylim=(-0.05,1.05))
    plot(fig, ax, V_ratio, eta, color=colors.dark_blue,linestyle='-',auto_scale=auto_scale,label=r'$\rm \eta$')
    plot(fig, ax, V_ratio, xi, color=colors.dark_red,linestyle='-',auto_scale=auto_scale,label=r'$\rm \xi$')
    plot(fig, ax, V_ratio, gamma, color=colors.light_blue,linestyle='-',auto_scale=auto_scale,label=r'$\rm \gamma$')
    plot(fig, ax, V_ratio, beta, color=colors.dark_orange,linestyle='-',auto_scale=auto_scale,label=r'$\rm \beta$')
    plot(fig, ax, V_ratio, lam, color=colors.dark_blue,linestyle=':',auto_scale=auto_scale,label=r'$\rm \lambda$')
    plot(fig, ax, V_ratio, alpha, color=colors.dark_red,linestyle=':',auto_scale=auto_scale,label=r'$\rm \alpha$')
    plt.savefig(fig_dir+'/'+imag_name+fmt, bbox_inches='tight', transparent=True, dpi=dpi)
    if I_in[n] == I_in[-1]:
        legend(fig,ax,auto_scale=auto_scale)
        plt.savefig(fig_dir+'/'+imag_name+'_legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

    # # Plot 1
    # imag_name = f'STREED_v1_formula_primes_I_in-{I_in[n]:0.0f}W'
    # title = r'Heat exchanger, $I_{\rm in}$ = '+f'{I_in[n]:0.0f} W'
    # xlabel = r'Flow rate ratio $V_{\rm air}/V_{\rm feed}$'
    # ylabel = r'Efficiency'
    # fig, ax = plot_setup(xlabel, ylabel,auto_scale=auto_scale, figsize=figsize, xlim=(100,1000),ylim=(-0.05,1.05))
    # plot(fig, ax, V_ratio, eta, color=colors.dark_blue,linestyle='-',auto_scale=auto_scale,label=r'$\rm \eta$')
    # plot(fig, ax, V_ratio, xi, color=colors.dark_red,linestyle='-',auto_scale=auto_scale,label=r'$\rm \xi$')
    # plot(fig, ax, V_ratio, eta_prime, color=colors.dark_blue,linestyle='-.',auto_scale=auto_scale,label=r'$\rm \eta^{\prime}$')
    # plot(fig, ax, V_ratio, xi_prime, color=colors.dark_red,linestyle='-.',auto_scale=auto_scale,label=r'$\rm \xi^{\prime}$')
    # plt.savefig(fig_dir+'/'+imag_name+fmt, bbox_inches='tight', transparent=True, dpi=dpi)
    # if I_in[n] == I_in[-1]:
    #     legend(fig,ax,auto_scale=auto_scale)
    #     plt.savefig(fig_dir+'/'+imag_name+'_legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

    # # Plot 3
    # imag_name = f'STREED_v1_formula_GORs_I_in-{I_in[n]:0.0f}W'
    # title = r'Heat exchanger, $I_{\rm in}$ = '+f'{I_in[n]:0.0f} W'
    # xlabel = r'Flow rate ratio $V_{\rm air}/V_{\rm feed}$'
    # ylabel = r'SWP (L/kWh)'
    # fig, ax = plot_setup(xlabel, ylabel,auto_scale=auto_scale, figsize=figsize, xlim=(100,1000))
    # plot(fig, ax, V_ratio, SWP, color='black',linestyle=':',linewidth=5,auto_scale=auto_scale,label=r'True SWP')
    # plot(fig, ax, V_ratio, LkWh*GOR1, color=colors.dark_green,linestyle='-',auto_scale=auto_scale,label=r'Original')
    # plot(fig, ax, V_ratio, LkWh*GOR2, color=colors.dark_purple,linestyle=':',linewidth=3,auto_scale=auto_scale,label=r'No $\alpha$')
    # plot(fig, ax, V_ratio, LkWh*GOR3, color=colors.dark_orange,linestyle='--',auto_scale=auto_scale,label=r'No $\lambda$')
    # plot(fig, ax, V_ratio, LkWh*GOR4, color=colors.dark_purple,linestyle='-.',linewidth=2.5,auto_scale=auto_scale,label=r'No $\alpha$, no $\lambda$')
    # plt.savefig(fig_dir+'/'+imag_name+fmt, bbox_inches='tight', transparent=True, dpi=dpi)
    # if I_in[n] == I_in[-1]:
    #     legend(fig,ax,auto_scale=auto_scale)
    #     plt.savefig(fig_dir+'/'+imag_name+'_legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

    # # Plot 4
    # imag_name = f'STREED_v1_formula_GORs_zoom_I_in-{I_in[n]:0.0f}W'
    # title = r'Heat exchanger, $I_{\rm in}$ = '+f'{I_in[n]:0.0f} W'
    # xlabel = r'Flow rate ratio $V_{\rm air}/V_{\rm feed}$'
    # ylabel = r'SWP (L/kWh)'
    # xrg = (250,600)
    # start = np.where(V_ratio == V_ratio[V_ratio>=xrg[0]][0])[0][0]
    # stop = np.where(V_ratio == V_ratio[V_ratio>xrg[1]][0])[0][0]-1
    # yrg = np.amax(SWP[start:stop])-np.amin(SWP[start:stop]) # SWP range
    # ymin = np.amin(SWP[start:stop])-0.05*yrg
    # ymax = np.amax(SWP[start:stop])+0.05*yrg
    # fig, ax = plot_setup(xlabel, ylabel,auto_scale=auto_scale, figsize=figsize, xlim=xrg, ylim=(ymin,ymax))
    # plot(fig, ax, V_ratio, SWP, color='black',linestyle=':',linewidth=5,auto_scale=auto_scale,label=r'True SWP')
    # plot(fig, ax, V_ratio, LkWh*GOR1, color=colors.dark_green,linestyle='-',linewidth=2,auto_scale=auto_scale,label=r'Original')
    # plot(fig, ax, V_ratio, LkWh*GOR2, color=colors.dark_purple,linestyle=':',linewidth=3.5,auto_scale=auto_scale,label=r'No $\alpha$')
    # plot(fig, ax, V_ratio, LkWh*GOR3, color=colors.dark_orange,linestyle='--',linewidth=2,auto_scale=auto_scale,label=r'No $\lambda$')
    # plot(fig, ax, V_ratio, LkWh*GOR4, color=colors.dark_purple,linestyle='-.',linewidth=3,auto_scale=auto_scale,label=r'No $\alpha$, no $\lambda$')
    # plt.savefig(fig_dir+'/'+imag_name+fmt, bbox_inches='tight', transparent=True, dpi=dpi)
    # if I_in[n] == I_in[-1]:
    #     legend(fig,ax,auto_scale=auto_scale)
    #     plt.savefig(fig_dir+'/'+imag_name+'_legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)


#%%