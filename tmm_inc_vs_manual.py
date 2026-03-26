# %%

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors
# %% ##############################################################################################

A = 10
gam = 0.0022
nb = 1.7

n_list, d_list = tmm_h.generate_n_and_d_new(gam, A, nb, delta=0.03, plot_flag=True, zoomed=False)

# %% #############################################################################################

lambda_list = np.linspace(2,5,100)
delta_lamb = lambda_list[-1] - lambda_list[0]
losses_total = np.sum(d_list * np.imag(n_list))
print(f'losses total is: {losses_total}')
trans_bulk = np.exp(-4*np.pi*losses_total/lambda_list)
emiss_bulk = 1 - trans_bulk

# add semi-infinite air layers
d_list.append(np.inf)
d_list.insert(0, np.inf)
n_list.append(nb)       
n_list.insert(0, nb)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_wavelength(n_list, d_list, lambda_list)
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_wavelength(n_list_reversed, d_list_reversed, lambda_list)

FOM_RL = np.trapz(T_list_RL, x=lambda_list)**2 / np.trapz(A_list_RL, x=lambda_list) / delta_lamb
FOM_LR = np.trapz(T_list_LR, x=lambda_list)**2 / np.trapz(A_list_LR, x=lambda_list) / delta_lamb
ASYM = np.trapz(A_list_LR, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list)
FOM_bulk = np.trapz(trans_bulk, x=lambda_list)**2 / np.trapz(emiss_bulk, x=lambda_list) / delta_lamb

print('TMM single KK stack')
print(f'FOM_RL: {FOM_RL}')
print(f'FOM_bulk: {FOM_bulk}')
print(f'FOM enhancement: {FOM_RL/FOM_bulk}')
print(f'ASYM: {ASYM}')
# %% ############################################################################################

# plotting using Will's plot modules

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
title = f'Transmittance (A={A}, x$_0$={gam}$\mu$m)'
#title = 'Reflectance or Absorbance'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

#plot(fig,ax,lambda_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

#plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
#plot(fig,ax,lambda_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)

# %%

I = []
N = 20
for i in range(len(lambda_list)):
    C = (1/T_list_LR[i]) * np.array([[1, -R_list_RL[i]], [R_list_LR[i], T_list_LR[i]**2 - R_list_LR[i]*R_list_RL[i]]])
    P = np.array([[1, 0], [0, 1]])
    stack = P @ C
    res = C
    for j in range(N-1):
        res = res @ stack
    I.append(res)

print(I[0])
T2, R2_LR, R2_RL, A2_LR, A2_RL = ([] for i in range(5))
for i in range(len(lambda_list)):
    I_i = I[i]
    R2_LR.append(I_i[1,0]/I_i[0,0])
    T2.append(1/I_i[0,0])
    R2_RL.append(-I_i[0,1]/I_i[0,0])
    A2_LR.append(1 - T2[i] - R2_LR[i])
    A2_RL.append(1 - T2[i] - R2_RL[i])

trans_bulk = np.exp(-4*np.pi*N*losses_total/lambda_list)
emiss_bulk = 1 - trans_bulk

FOM_RL = np.trapz(T2, x=lambda_list)**2 / np.trapz(A2_RL, x=lambda_list) / delta_lamb
FOM_LR = np.trapz(T2, x=lambda_list)**2 / np.trapz(A2_LR, x=lambda_list) / delta_lamb
ASYM = np.trapz(A2_LR, x=lambda_list) / np.trapz(A2_RL, x=lambda_list)
FOM_bulk = np.trapz(trans_bulk, x=lambda_list)**2 / np.trapz(emiss_bulk, x=lambda_list) / delta_lamb

print(f'TMM {N} stacks')
print(f'FOM_RL: {FOM_RL}')
print(f'FOM_bulk: {FOM_bulk}')
print(f'FOM enhancement: {FOM_RL/FOM_bulk}')
print(f'ASYM: {ASYM}')

# %%
def generate_StackofStacks(gam, a, nb, num_stacks, t_prop):
    n_list = []
    d_list = []
    c_list = []
    c_list_KK = []

    n_list_KK, d_list_KK = tmm_h.generate_n_and_d_new(gam, a, nb, delta=0.03, plot_flag=False)
    t_KK = np.sum(d_list_KK)
    t_inc = t_prop*t_KK
    print(f"t_KK is: {t_KK}")
    print(f"t_inc is: {t_inc}")
    t_total = num_stacks*(t_inc + t_KK)
    print(f"t_total is: {t_total}")
    #print(f'length of KK MS is: {t_KK}')

    k_bulk = np.sum(d_list_KK * np.imag(n_list_KK)) / t_KK
    print(f'k_bulk from np.sum is: {k_bulk}')
    losses_total = t_total * k_bulk
    print(f'total losses are: {losses_total}')

    for i in range(len(d_list_KK)):
        c_list_KK.append('c')

    for i in range(num_stacks):
        n_list.extend(n_list_KK)
        d_list.extend(d_list_KK)
        c_list.extend(c_list_KK)
        n_list.append(nb)
        d_list.append(t_inc)
        c_list.append('i')
    
    d_list.append(np.inf)
    d_list.insert(0, np.inf)
    n_list.append(nb)
    n_list.insert(0, nb)
    c_list.insert(0, 'i')
    c_list.append('i')


    return (n_list, d_list, c_list)

#%% ###

num_stacks = 20
t_prop = 0

n_list, d_list, c_list = generate_StackofStacks(gam, A, nb, num_stacks, t_prop)

n_list_reversed = n_list[::-1]
d_list_reversed = d_list[::-1]
c_list_reversed = c_list[::-1]

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_wavelength_inc(n_list, d_list, c_list, lambda_list)
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_wavelength_inc(n_list_reversed, d_list_reversed, c_list_reversed, lambda_list)

# %%
xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
#title = f'TRA (A={A}, x$_0$={gam}$\mu$m)'
title = f'Transmittance for {num_stacks} stacks'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,T2,label=r'T, manual',color=colors.blue,auto_scale=True)
plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T, tmm',color=colors.light_blue,auto_scale=True)

legend(fig,ax,auto_scale=True)

title = f'Absorbance for {num_stacks} stacks'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)
plot(fig,ax,lambda_list,A2_RL, label=r'A$_{{RL}}$, manual',color=colors.blue,auto_scale=True)
plot(fig,ax,lambda_list,A2_LR, label=r'A$_{{LR}}$, manual',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,A_list_RL, '--', label=r'A$_{{RL}}$, tmm',color=colors.light_blue,auto_scale=True)
plot(fig,ax,lambda_list,A_list_LR, '--', label=r'A$_{{LR}}$, tmm',color=colors.light_red,auto_scale=True)
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)
legend(fig,ax,auto_scale=True)

title = f'Reflectance for {num_stacks} stacks'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,R2_RL, label=r'R$_{{RL}}$, manual',color=colors.blue,auto_scale=True)
plot(fig,ax,lambda_list,R2_LR, label=r'R$_{{LR}}$, manual',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,R_list_RL, '--', label=r'R$_{{RL}}$, tmm',color=colors.light_blue,auto_scale=True)
plot(fig,ax,lambda_list,R_list_LR, '--', label=r'R$_{{LR}}$, tmm',color=colors.light_red,auto_scale=True)


legend(fig,ax,auto_scale=True)
# %%

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Error'
#title = f'Transmittance (A={A}, x$_0$={gam}$\mu$m)'
title = f'TRA Error between TMM and Manual ({num_stacks} stacks)'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,(T2 - T_list_LR)/T2,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,lambda_list,(A2_RL - A_list_RL)/A2_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,(A2_LR - A_list_LR)/A2_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,lambda_list,(R2_RL - R_list_RL)/R2_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,(R2_LR - R_list_LR)/R2_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)
legend(fig,ax,auto_scale=True)

# %%
