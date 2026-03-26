# %%

import tmm
import tmm_helper as tmm_h
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from plot_functions import plot_setup, plot, legend
import colors
# %% ##############################################################################################

A = 10
gam = 0.05
nb = 2.3
delta_arr = np.arange(0.005, 0.5, 0.001)
R_arr = []
lamb = 3

for i, delta in enumerate(delta_arr):
    print(delta)
    n_list, d_list = tmm_h.generate_n_and_d_new(gam, A, nb, delta=delta, plot_flag=False)

    d_list.append(np.inf)
    d_list.insert(0, np.inf)
    n_list.append(nb)       
    n_list.insert(0, nb)

    T_LR, R_LR, A_LR = tmm_h.TRA(n_list, d_list, lamb)
    R_arr.append(R_LR)

# %%

xlabel = 'delta'; ylabel = 'Reflectance'
title = f'Reflectance (A={A}, x$_0$={gam}$\mu$m)'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(np.log10(0.005),np.log10(0.5)),figsize=(5,4),auto_scale=True)

plot(fig,ax,np.log10(delta_arr),np.log10(R_arr),label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

# %%

np.savetxt(f'C:\\Users\\kl89\\MS Window Project\\Data\\R_vs_delta~0.005-0.5_A~{A}_gam~{gam}_nb~{nb}_L~300gam.txt', R_arr)

# %%

R_1 = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\Data\\R_vs_delta~0.005-0.5_A~10_gam~0.05_nb~2.3.txt')
R_8 = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\Data\\R_vs_delta~0.005-0.5_A~10_gam~0.05_nb~2.3_L~380gam.txt')
R_10 = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\Data\\R_vs_delta~0.005-0.5_A~10_gam~0.05_nb~2.3_L~350gam.txt')
R_50 = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\Data\\R_vs_delta~0.005-0.5_A~10_gam~0.05_nb~2.3_L~300gam.txt')

xlabel = 'delta'; ylabel = 'Reflectance'
title = f'Reflectance'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(np.log10(0.005),np.log10(0.5)),figsize=(5,4),auto_scale=True)

plot(fig,ax,np.log10(delta_arr),np.log10(R_1),label='L=400gam', color=colors.blue,auto_scale=True)
plot(fig,ax,np.log10(delta_arr),np.log10(R_8),label='L=380gam',color=colors.red,auto_scale=True)
plot(fig,ax,np.log10(delta_arr),np.log10(R_10),label='L=350gam',color=colors.green,auto_scale=True)
plot(fig,ax,np.log10(delta_arr),np.log10(R_50),label='L=300gam',color=colors.purple,auto_scale=True)

legend(fig,ax,auto_scale=True)
# %%

print(delta_arr[71])
print(delta_arr[111])

# %%

A = 10
gam = 0.05
nb = 2.3
#delta_arr = np.arange(0.005, 0.5, 0.001)
delta_arr = np.logspace(-2.3, -0.3, 100)  # 0.005 … 0.5
R_arr = []
lamb = 3

R_arr = []
E_rms_arr = []

for i, delta in enumerate(delta_arr):
    print(delta)
    n_list, d_list = tmm_h.generate_n_and_d_new(gam, A, nb, delta=delta, plot_flag=False)

    d_list.append(np.inf)
    d_list.insert(0, np.inf)
    n_list.append(nb)       
    n_list.insert(0, nb)

    #T_LR, R_LR, A_LR = tmm_h.TRA(n_list, d_list, lamb)
    r = tmm.coh_tmm('p', n_list, d_list, 0, lamb)['r']
    #R_arr.append(R_LR)
    R_arr.append(abs(r))

    n_list, d_list = tmm_h.generate_n_and_d_new(gam, A, nb, delta=delta, plot_flag=False)
    n_list = np.array(n_list, dtype=np.complex128)
    d_list = np.array(d_list)          # physical thicknesses (µm)

    # --- 2. build a piece-wise-constant 1-D grid ----------------------
    # use two points per layer → stair-step profile
    edges   = np.concatenate(([0.0], np.cumsum(d_list)))    # 0 … L
    x_dense = np.repeat(edges, 2)[1:-1]                     # mid/edge alternation
    n_dense = np.repeat(n_list, 2)                          # step const.

    dx = np.diff(x_dense).mean()        # uniform spacing now
    N  = len(x_dense)

    # --- zero-pad 20 % of the points on each side --------------------
    pad = int(0.2*N)
    n_im_padded = np.concatenate([np.zeros(pad), n_dense.imag, np.zeros(pad)])

    # FFT frequencies correspond to the *padded* array
    k  = fftfreq(n_im_padded.size, d=dx)
    Hk = -1j*np.sign(k)

    n_re_pred_padded = ifft(fft(n_im_padded) * Hk).real
    n_re_pred = n_re_pred_padded[pad:-pad]      # drop pads to original length

    # mask: core region ±6γ
    mask = np.abs(x_dense - x_dense[N//2]) <= 6*gam

    n_scale = np.max(np.abs(n_dense.real - nb))
    diff = (n_dense.real[mask] - n_re_pred[mask]) / n_scale
    diff -= diff.mean()                       # remove DC level
    E_rms = np.sqrt(np.mean(diff**2))
    #E_rms = np.max(np.abs(diff))
    E_rms_arr.append(E_rms)

# %%
xlabel = 'delta'; ylabel = 'Reflectance'
title = f'Reflectance (A={A}, x$_0$={gam}$\mu$m)'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(np.log10(0.005),np.log10(0.5)),figsize=(5,4),auto_scale=True)

plot(fig,ax,np.log10(delta_arr),np.log10(R_arr),label=r'T',color=colors.blue,auto_scale=True)

# %%
xlabel = 'delta'; ylabel = 'Hilbert RMS Error'
title = f'RMS Error (A={A}, x$_0$={gam}$\mu$m)'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(np.log10(0.005),np.log10(0.5)),figsize=(5,4),auto_scale=True)

plot(fig,ax,np.log10(delta_arr),E_rms_arr,label=r'T',color=colors.blue,auto_scale=True)

# %%

E_floor = E_rms_arr[0]
E_rel   = abs(E_rms_arr - E_floor) / E_floor

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.loglog(delta_arr, R_arr, 'g',)
ax2.loglog(delta_arr, E_rel, 'b')

ax1.set_xlabel('δ (log scale)')
ax1.set_ylabel('abs reflection coefficient', color='g')
ax2.set_ylabel('Hilbert RMS error', color='b')

# %%
