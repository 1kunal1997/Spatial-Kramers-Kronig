#%%###########################################################
# Libraries and settings
##############################################################
# Libraries 
import os
import numpy as np
import matplotlib.pyplot as plt

# Reset plot style to default
plt.close('all') # close all open figures
plt.style.use('default') # Set to default style for viewing ease in interactive window

# Directories
script_dir = os.path.dirname(__file__) # automatically find current directory

# Figure subfolder directory (create if it does not exist)
fig_dir = os.path.join(script_dir, 'Figures/') # figures directory
# if not os.path.isdir(fig_dir):
#     os.makedirs(fig_dir)

# Load external plotting functions
from plot_functions import plot_setup, plot, legend

# Load plotting colors
import colors # make available colors from schmid_colors.py 

# Image file settings
fmt = '.png' # image format (use png for PowerPoint, pdf and eps for publications)
dpi = 300 # image resolution, density of pixels per inch (use at least 300)


#%%###########################################################
# Load data from COMSOL 
##############################################################
# # Directory with COMSOL data (.csv exported from Global Evaluation Group in COMSOL)
# data_dir = os.path.join(script_dir, 'Data/') # data directory

# # COMSOL data filename
# filename = 'filename.csv'

# # Import COMSOL data as array
# data = np.genfromtxt(data_dir+filename,delimiter=',',skip_header=5) # load data from csv 

# # Column indices of plotted variables (variable on nth column accesed via data[:,n])
# xind = 0 # x variable (0th column)
# yind = 1 # y variable (1st column)


#%%###########################################################
# Plot and save figure
##############################################################
savename = 'matplotlib_template_figure'
xlabel = 'x'; ylabel = 'y'
fig,ax = plot_setup(xlabel,ylabel,figsize=(5,4),auto_scale=True) # set up axis (try different fig sizes/aspect ratios)
# plot(ax,data[:,xind],data[:,yind])
# plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=True,dpi=dpi) # save figure as image file

# DELETE THIS BLOCK IF YOU ARE PLOTTING COMSOL DATA ###########################################
x = np.arange(-2,2+0.05,0.05) # x ranging from -2 to 2 in steps of 0.05
plot(fig,ax,x,x**2,label=r'$y = x^2$',color=colors.green,auto_scale=True) # plot x^2 in green
plot(fig,ax,x,x**3,label=r'$y = x^3$',color=colors.purple,auto_scale=True) # plot x^3 in purple
###############################################################################################

# Create and save legend for above figure
legend(fig,ax,auto_scale=True) # create legend from curves labeled above
# plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi) # save legend as image file

#%%