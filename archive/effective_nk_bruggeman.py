# In[1]:

import numpy as np
import matplotlib.pyplot as plt

# Load external plotting functions
from plot_functions import plot_setup, plot, legend

# Load plotting colors
import colors # make available colors from schmid_colors.py 

# Image file settings
fmt = '.png' # image format (use png for PowerPoint, pdf and eps for publications)
dpi = 300 # image resolution, density of pixels per inch (use at least 300)
fig_dir = 'C:\\Users\\kl89\\MS Window Project\\Figures\\Squared Off n and k\\'

# %%
def n_eff_bruggeman(n1, c1, n2, c2):
    e1 = np.square(n1)
    e2 = np.square(n2)

    Hb = e1*(3*c1 - 1) + e2*(3*c2 - 1)

    return (np.sqrt((Hb - np.sqrt(np.square(Hb) + 8*e1*e2)) / 4), np.sqrt((Hb + np.sqrt(np.square(Hb) + 8*e1*e2)) / 4) )

# In[2]:

n1 = 1.7
c1 = 0.50
#n2 = 1.01+22.26*1j
n2 = 1 + 26*1j
c2 = 1 - c1

n_eff = n_eff_bruggeman(n1, c1, n2, c2)

print(n_eff)

# %%
n1 = 1.48
n2 = 4.9 + 5.4*1j
c2_arr = np.arange(0.01, 0.99, 0.01)
n_eff_arr = np.zeros((len(c2_arr), 2), dtype=complex)

for i, c2 in enumerate(c2_arr):
    c1 = 1 - c2
    n_eff_arr[i] = (n_eff_bruggeman(n1, c1, n2, c2))

print(n_eff_arr[:][0])
plt.plot(c2_arr, np.real(n_eff_arr[:,0]), label='real 1')
plt.plot(c2_arr, np.imag(n_eff_arr[:,0]), label='imag 1')
plt.plot(c2_arr, np.real(n_eff_arr[:,1]), '--', label='real 2')
plt.plot(c2_arr, np.imag(n_eff_arr[:,1]), '--', label='imag 2')
plt.legend(loc='best')
plt.xlabel('Relative volume of metal')
plt.ylabel('Refractive Index')
# %%

plt.semilogy(c2_arr, np.imag(n_eff_arr))
# %%
n1 = 1.9
n2 = 8 + 10*1j
c2_arr = np.arange(0.01, 0.99, 0.01)
n_eff_arr = np.zeros((len(c2_arr), 2), dtype=complex)

for i, c2 in enumerate(c2_arr):
    c1 = 1 - c2
    n_eff_arr[i] = (n_eff_bruggeman(n1, c1, n2, c2))

savename = f''
xlabel = 'Relvative Volume of Metal'; ylabel = 'Refractive Index'
title = f"Effective RI of Yttria and n$_2$ = {n2}"
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(c2_arr[0],c2_arr[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,c2_arr,np.real(n_eff_arr[:,0]), label='real (-)',color=colors.red,auto_scale=True)
plot(fig,ax,c2_arr,np.imag(n_eff_arr[:,0]), label='imag (-)',color=colors.purple,auto_scale=True)

plot(fig,ax,c2_arr,np.real(n_eff_arr[:,1]), '--', label='real (+)',color=colors.light_red,auto_scale=True)
plot(fig,ax,c2_arr,np.imag(n_eff_arr[:,1]), '--', label='imag (+)',color=colors.light_purple,auto_scale=True)

#plt.savefig(fig_dir+savename+fmt,bbox_inches='tight',transparent=False,dpi=dpi)
legend(fig,ax,auto_scale=True)

# %%
