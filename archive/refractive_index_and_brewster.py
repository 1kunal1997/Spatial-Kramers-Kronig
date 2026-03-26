#%%
import numpy as np
from refractiveindex import RefractiveIndexMaterial
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
# Load external plotting functions
from plot_functions import plot_setup, plot, legend

# Load plotting colors
import colors # make available colors from schmid_colors.py 

# Image file settings
fmt = '.png' # image format (use png for PowerPoint, pdf and eps for publications)
dpi = 300 # image resolution, density of pixels per inch (use at least 300)
fig_dir = 'C:\\Users\\kl89\\MS Window Project\\Figures\\'
um = 1e3
deg = np.pi / 180
wls = np.linspace(2,5,50) * um
materials = ["SiO2" , "Y2O3" , "Al2O3" , "MgO" , "MgF2" , "ZnSe" , "ZnS" , "Si" , "Ge"]

#%%
page_lookup = {
    "SiO2":"Franta",    # "Franta" , "Malitson" , "Kischkat"
    "Y2O3":"Nigara",    # "Nigara"
    "Al2O3":"Malitson", # "Malitson" , "Querry" , "Boidin"
    "MgO":"Stephens",   # Stephens"
    "MgF2":"Franta",    # "Franta" , "Dodge" , "Li" , "Zheng"
    "ZnSe":"Querry",    # "Querry" , "Connolly" , "Amotchkina"
    "ZnS":"Debenham",   # "Debenham" , "Querry"
    "Si":"Franta",      # "Franta" , "Li"
    "Ge":"Amotchkina",  # "Burnett" , "Li"
    "Ni-Fe":"Tikuisis_bare150nm",
}

for material in materials:
  author = page_lookup[material]
  tmp = RefractiveIndexMaterial(shelf="main" , book=material , page=author) # This defines the material
  n_tmp = [tmp.get_refractive_index(wl) for wl in wls]                      # This extracts n of that material, here I do it in a loop over wavelengths
  plt.figure(1)
  plt.plot(wls/um , n_tmp , label = material+' - '+author)
  try:
    k_tmp = [tmp.get_extinction_coefficient(wl) for wl in wls]  
    plt.figure(2)
    plt.plot(wls/um , k_tmp , label = material)    
  except:
    continue

wls_new = np.linspace(0.2, 1.65, 50) * um
permalloy = RefractiveIndexMaterial(shelf="other" , book='Ni-Fe' , page="Tikuisis_bare150nm")
n_tmp = [permalloy.get_refractive_index(wl) for wl in wls_new] 
k_tmp = [permalloy.get_extinction_coefficient(wl) for wl in wls_new]
n_interp_linear = interp1d(wls_new, n_tmp, kind='linear', fill_value='extrapolate')
n_interp_quad = interp1d(wls_new, n_tmp, kind='quadratic', fill_value='extrapolate')
k_interp_linear = interp1d(wls_new, k_tmp, kind='linear', fill_value='extrapolate')
k_interp_quad = interp1d(wls_new, k_tmp, kind='quadratic', fill_value='extrapolate')

def l( x, a, b, c, d ):
    return a*np.log( b*x + c ) + d
wls_newer = np.linspace(0.2, 5) * um
n_param = curve_fit( l, wls_new, n_tmp)
k_param = curve_fit( l, wls_new, k_tmp)
n_coeffs = n_param[0]
k_coeffs = k_param[0]

savename = f'NiFe_ref_real_extrapolate_0.2-5um'
title = f'Real Refractive Index of Ni-Fe'
xlabel = "Wavelength ($\mu$m)"; ylabel = "refractive index"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax, wls_new/um, n_tmp, '*', markersize=6, label='data', color=colors.blue,auto_scale=True)
plot(fig,ax, wls_newer/um , n_interp_linear(wls_newer), label='linear', color=colors.red,auto_scale=True)
plot(fig,ax, wls_newer/um, n_interp_quad(wls_newer), label='quad', color=colors.green,auto_scale=True)
plot(fig,ax, wls_newer/um, l(wls_newer, *n_coeffs), label='log', color=colors.purple,auto_scale=True)

plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
legend(fig,ax,auto_scale=True) # create legend from curves labeled above
plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

savename = f'NiFe_ref_imag_extrapolate_0.2-5um'
title = f'Imaginary Refractive Index of Ni-Fe'
xlabel = "Wavelength ($\mu$m)"; ylabel = "refractive index"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax, wls_new/um, k_tmp, '*', markersize=6, label='data', color=colors.blue,auto_scale=True)
plot(fig,ax, wls_newer/um , k_interp_linear(wls_newer), label='linear', color=colors.red,auto_scale=True)
plot(fig,ax, wls_newer/um, k_interp_quad(wls_newer), label='quad', color=colors.green,auto_scale=True)
plot(fig,ax, wls_newer/um, l(wls_newer, *k_coeffs), label='log', color=colors.purple,auto_scale=True)

plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
legend(fig,ax,auto_scale=True) # create legend from curves labeled above
plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

savename = f'NiFe_ref_data_0.2-1.6um'
title = f'Refractive Index Data of Ni-Fe'
xlabel = "Wavelength ($\mu$m)"; ylabel = "refractive index"
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True)

plot(fig,ax, wls_new/um, n_tmp, '*', markersize=6, label='real', color=colors.blue,auto_scale=True)
plot(fig,ax, wls_new/um, k_tmp, '*', markersize=6, label='imag', color=colors.red,auto_scale=True)

plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
legend(fig,ax,auto_scale=True) # create legend from curves labeled above
plt.savefig(fig_dir+savename+'legend'+fmt, bbox_inches='tight', transparent=True, dpi=dpi)

plt.figure(1)
plt.xlabel("Wavelength (microns)")
plt.ylabel("Real refractive index")
plt.legend(loc='best')

plt.figure(2)
plt.xlabel("Wavelength (microns)")
plt.ylabel("Imaginary Refractive Index")
plt.legend(loc='best')
# %%
