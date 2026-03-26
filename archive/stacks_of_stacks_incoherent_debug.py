# %%

import tmm_helper as tmm_h
import numpy as np
from plot_functions import plot_setup, plot, legend
import colors
# %% ##############################################################################################

A = 5
gam = 0.0022
nb = 1.7

#n_list, d_list = tmm_h.generate_n_and_d(gam, A, nb, min_thickness=0.0001, plot_flag=True)
n_list, d_list = tmm_h.generate_n_and_d_new(gam, A, nb, delta=0.03, plot_flag=False)

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
#title = f'Transmittance (A={A}, x$_0$={gam}$\mu$m)'
title = 'Reflectance or Absorbance'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,lambda_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,lambda_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)

# %%

I = []
N = 2
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
xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
title = f'TRA (A={A}, x$_0$={gam}$\mu$m)'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,lambda_list,T2,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,lambda_list,A2_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,A2_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

#plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
#plot(fig,ax,lambda_list,R2, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)
# %%######################################################################################
#losses_total = 0.0046053     #0.004614
N = 10000
rows, cols = (len(lambda_list), N)
I = np.empty((len(lambda_list), N), dtype=object)

for i in range(len(lambda_list)):
    C = (1/T_list_LR[i]) * np.array([[1, -R_list_RL[i]], [R_list_LR[i], T_list_LR[i]**2 - R_list_LR[i]*R_list_RL[i]]])
    P = np.array([[1, 0], [0, 1]])
    stack = P @ C
    res = C
    for j in range(N):
        I[i][j] = res
        res = res @ stack

# %% ####################################################################################

T2, R2_LR, R2_RL, A2_LR, A2_RL, emiss_bulk, trans_bulk = (np.empty((len(lambda_list), N), dtype=float) for i in range(7))
FOM_RL, FOM_LR, ASYM, FOM_bulk, FOM_enh = (np.empty(N, dtype=float) for i in range(5))

for i in range(len(lambda_list)):
    for j in range(N):
        I_i = I[i][j]
        R2_LR[i][j] = (I_i[1,0]/I_i[0,0])
        T2[i][j] = (1/I_i[0,0])
        R2_RL[i][j] = (-I_i[0,1]/I_i[0,0])
        A2_LR[i][j] = (1 - T2[i][j] - R2_LR[i][j])
        A2_RL[i][j] = (1 - T2[i][j] - R2_RL[i][j])

for j in range(N):
    trans_bulk[:,j] = np.exp(-4*np.pi*(j+1)*losses_total/lambda_list)
    emiss_bulk[:,j] = 1 - trans_bulk[:,j]
    FOM_RL[j] = np.trapz(T2[:,j], x=lambda_list)**2 / np.trapz(A2_RL[:,j], x=lambda_list) / delta_lamb
    FOM_LR[j] = np.trapz(T2[:,j], x=lambda_list)**2 / np.trapz(A2_LR[:,j], x=lambda_list) / delta_lamb
    ASYM[j] = np.trapz(A2_LR[:,j], x=lambda_list) / np.trapz(A2_RL[:,j], x=lambda_list)
    FOM_bulk[j] = np.trapz(trans_bulk[:,j], x=lambda_list)**2 / np.trapz(emiss_bulk[:,j], x=lambda_list) / delta_lamb
    FOM_enh[j] = FOM_RL[j] / FOM_bulk[j]

print(losses_total)
# %%
xlabel = 'Number of Stacks'; ylabel = 'FOM enhancement'
title = f'A = {A}, x$_0$ = {gam}'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(0,N),figsize=(5,4),auto_scale=True)

plot(fig,ax,range(N),FOM_enh, color=colors.blue, label='KK',auto_scale=True)
#plot(fig,ax,range(N),np.log(FOM_bulk), '--', color=colors.red, label='bulk',auto_scale=True)

legend(fig,ax,auto_scale=True)

# %%
np.savetxt(f'C:\\Users\\kl89\\MS Window Project\\Data\\FOM_enh_vs_numStacks~{N}_A~{A}_gam~{gam}.txt', FOM_enh)

# %%
A_01 = np.loadtxt(f'C:\\Users\\kl89\\MS Window Project\\Data\\FOM_enh_vs_numStacks~100000_A~0.1_gam~0.002.txt')

xlabel = 'Number of Stacks'; ylabel = 'FOM enhancement'
title = f'A = {0.1}, x$_0$ = {gam}'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(0,100000),figsize=(5,4),auto_scale=True)

plot(fig,ax,range(100000),A_01, color=colors.light_blue, label='KK',auto_scale=True)

# %%

A_3 = np.loadtxt(f'C:\\Users\\kl89\\MS Window Project\\Data\\FOM_enh_vs_numStacks~10000_A~5_gam~0.0007.txt')
A_4 = np.loadtxt(f'C:\\Users\\kl89\\MS Window Project\\Data\\FOM_enh_vs_numStacks~10000_A~5_gam~0.0009.txt')
A_5 = np.loadtxt(f'C:\\Users\\kl89\\MS Window Project\\Data\\FOM_enh_vs_numStacks~10000_A~5_gam~0.001.txt')
A_6 = np.loadtxt(f'C:\\Users\\kl89\\MS Window Project\\Data\\FOM_enh_vs_numStacks~10000_A~5_gam~0.002.txt')
A_7 = np.loadtxt(f'C:\\Users\\kl89\\MS Window Project\\Data\\FOM_enh_vs_numStacks~10000_A~5_gam~0.0022.txt')

xlabel = 'Number of Stacks'; ylabel = 'FOM enhancement'
title = f'A=5, n$_b$=1.7'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(0,N),figsize=(5,4),auto_scale=True)

plot(fig,ax,range(N),A_3, color=colors.blue, label='x$_0$=0.7nm',auto_scale=True)
plot(fig,ax,range(N),A_4, color=colors.green, label='x$_0$=0.9nm',auto_scale=True)
plot(fig,ax,range(N),A_5, color=colors.red, label='x$_0$=1.0nm',auto_scale=True)
plot(fig,ax,range(N),A_6, color=colors.purple, label='x$_0$=2.0nm',auto_scale=True)
plot(fig,ax,range(N),A_7, color=colors.copper, label='x$_0$=2.2nm',auto_scale=True)
#plot(fig,ax,range(N),np.log(FOM_bulk), '--', color=colors.red, label='bulk',auto_scale=True)

legend(fig,ax,auto_scale=True)
# %%
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Choose a colormap (here 'Blues' for your blue gradient example)
cmap = cm.get_cmap('Blues', 5)
#sliced_cmap = lambda x: cmap(0.2 + 0.8*x)

params = [0.0007, 0.0009, 0.001, 0.002, 0.0022]
#params = [3, 4, 5, 6, 7]
y_vals = [A_3, A_4, A_5, A_6, A_7]
# Normalize the parameter values to [0,1] for the colormap
norm = mcolors.Normalize(vmin=params[0], vmax=params[-1])

xlabel = 'Number of Stacks'; ylabel = 'FOM enhancement'
title = f''
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(0,N),figsize=(5,4),auto_scale=True)

for i, p in enumerate(params):
    color = cmap(norm(p))  # Map param value to a color
    plot(fig,ax,range(N),y_vals[i], color=color, label=f'A={p}',auto_scale=True)

legend(fig,ax,auto_scale=True)
# %%

TRA_LR_Comsol = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\Data\\TRA_A~5_gam~0.001_LR.txt', skiprows=5)
TRA_RL_Comsol = np.loadtxt('C:\\Users\\kl89\\MS Window Project\\Data\\TRA_A~5_gam~0.001_RL.txt', skiprows=5)
lam_comsol = TRA_LR_Comsol[:,0]
trans_LR_comsol = TRA_LR_Comsol[:,1]
ref_LR_comsol = TRA_LR_Comsol[:,2]
abs_LR_comsol = TRA_LR_Comsol[:,3]
trans_RL_comsol = TRA_RL_Comsol[:,1]
ref_RL_comsol = TRA_RL_Comsol[:,2]
abs_RL_comsol = TRA_RL_Comsol[:,3]

# %% ############################################################################################

# plotting using Will's plot modules

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
#title = f'Transmittance (A={A}, x$_0$={gam}$\mu$m)'
title = 'Reflectance or Absorbance'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,lambda_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,lambda_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)

# %% #############################################################################################

#losses_total = 0.0046053
losses_total = 0.004614
trans_bulk = np.exp(-4*np.pi*losses_total/lam_comsol)
emiss_bulk = 1 - trans_bulk

delta_lamb_comsol = lam_comsol[-1] - lam_comsol[0]
FOM_RL_comsol = np.trapz(trans_RL_comsol, x=lam_comsol)**2 / np.trapz(abs_RL_comsol, x=lam_comsol) / delta_lamb_comsol
FOM_LR_comsol = np.trapz(trans_LR_comsol, x=lam_comsol)**2 / np.trapz(abs_LR_comsol, x=lam_comsol) / delta_lamb_comsol
FOM_bulk_comsol = np.trapz(trans_bulk, x=lam_comsol)**2 / np.trapz(emiss_bulk, x=lam_comsol) / delta_lamb_comsol
ASYM = np.trapz(abs_LR_comsol, x=lam_comsol) / np.trapz(abs_RL_comsol, x=lam_comsol)

print('Comsol single KK stack')
print(f'FOM_RL: {FOM_RL_comsol}')
print(f'FOM_bulk: {FOM_bulk_comsol}')
print(f'FOM enhancement: {FOM_RL_comsol/FOM_bulk_comsol}')
print(f'ASYM: {ASYM}')

# %%

I = []
N = 1000
for i in range(len(lam_comsol)):
    C = (1/trans_LR_comsol[i]) * np.array([[1, -ref_RL_comsol[i]], [ref_LR_comsol[i], trans_LR_comsol[i]**2 - ref_LR_comsol[i]*ref_RL_comsol[i]]])
    P = np.array([[1, 0], [0, 1]])
    stack = P @ C
    res = C
    for j in range(N-1):
        res = res @ stack
    I.append(res)

print(I[0])
T2, R2_LR, R2_RL, A2_LR, A2_RL = ([] for i in range(5))
for i in range(len(lam_comsol)):
    I_i = I[i]
    R2_LR.append(I_i[1,0]/I_i[0,0])
    T2.append(1/I_i[0,0])
    R2_RL.append(-I_i[0,1]/I_i[0,0])
    A2_LR.append(1 - T2[i] - R2_LR[i])
    A2_RL.append(1 - T2[i] - R2_RL[i])

trans_bulk = np.exp(-4*np.pi*N*losses_total/lam_comsol)
emiss_bulk = 1 - trans_bulk

FOM_RL = np.trapz(T2, x=lam_comsol)**2 / np.trapz(A2_RL, x=lam_comsol) / delta_lamb_comsol
FOM_LR = np.trapz(T2, x=lam_comsol)**2 / np.trapz(A2_LR, x=lam_comsol) / delta_lamb_comsol
FOM_bulk_comsol = np.trapz(trans_bulk, x=lam_comsol)**2 / np.trapz(emiss_bulk, x=lam_comsol) / delta_lamb_comsol
ASYM = np.trapz(A2_LR, x=lam_comsol) / np.trapz(A2_RL, x=lam_comsol)
print(f'Comsol {N} stacks')
print(f'FOM_RL: {FOM_RL}')
print(f'FOM_bulk: {FOM_bulk_comsol}')
print(f'FOM enhancement: {FOM_RL/FOM_bulk_comsol}')
print(f'ASYM: {ASYM}')

# %%

N = 1000
I = np.empty((len(lam_comsol), N), dtype=object)

for i in range(len(lam_comsol)):
    C = (1/trans_LR_comsol[i]) * np.array([[1, -ref_RL_comsol[i]], [ref_LR_comsol[i], trans_LR_comsol[i]**2 - ref_LR_comsol[i]*ref_RL_comsol[i]]])
    P = np.array([[1, 0], [0, 1]])
    stack = P @ C
    res = C
    for j in range(N):
        I[i][j] = res
        res = res @ stack

# %%

T2, R2_LR, R2_RL, A2_LR, A2_RL, emiss_bulk, trans_bulk = (np.empty((len(lam_comsol), N), dtype=float) for i in range(7))
FOM_RL, FOM_LR, ASYM, FOM_bulk, FOM_enh = (np.empty(N, dtype=float) for i in range(5))

for i in range(len(lam_comsol)):
    for j in range(N):
        I_i = I[i][j]
        R2_LR[i][j] = (I_i[1,0]/I_i[0,0])
        T2[i][j] = (1/I_i[0,0])
        R2_RL[i][j] = (-I_i[0,1]/I_i[0,0])
        A2_LR[i][j] = (1 - T2[i][j] - R2_LR[i][j])
        A2_RL[i][j] = (1 - T2[i][j] - R2_RL[i][j])

for j in range(N):
    trans_bulk[:,j] = np.exp(-4*np.pi*(j+1)*losses_total/lam_comsol)
    emiss_bulk[:,j] = 1 - trans_bulk[:,j]
    FOM_RL[j] = np.trapz(T2[:,j], x=lam_comsol)**2 / np.trapz(A2_RL[:,j], x=lam_comsol) / delta_lamb_comsol
    FOM_LR[j] = np.trapz(T2[:,j], x=lam_comsol)**2 / np.trapz(A2_LR[:,j], x=lam_comsol) / delta_lamb_comsol
    ASYM[j] = np.trapz(A2_LR[:,j], x=lam_comsol) / np.trapz(A2_RL[:,j], x=lam_comsol)
    FOM_bulk[j] = np.trapz(trans_bulk[:,j], x=lam_comsol)**2 / np.trapz(emiss_bulk[:,j], x=lam_comsol) / delta_lamb_comsol
    FOM_enh[j] = FOM_RL[j] / FOM_bulk[j]

print(FOM_RL[999]/FOM_bulk[999])
print(FOM_bulk[999])
print(FOM_RL[999])
# %%
xlabel = 'Number of Stacks'; ylabel = 'FOM (Comsol)'
title = f''
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(0,N),figsize=(5,4),auto_scale=True)

#plot(fig,ax,range(N),np.log(FOM_RL), color=colors.blue, label='KK',auto_scale=True)
#plot(fig,ax,range(N),np.log(FOM_bulk), '--', color=colors.red, label='bulk',auto_scale=True)
#plot(fig,ax,range(N),FOM_RL/FOM_LR, '--', color=colors.red, label='bulk',auto_scale=True)
plot(fig,ax,range(N),FOM_enh, color=colors.red, label='bulk',auto_scale=True)

legend(fig,ax,auto_scale=True)

print(FOM_RL[800])
print(FOM_bulk[800])
# %%

print(2.0131e-10/1.991e-10)
#%% ###########################################################################################################

# all code below this point is to verify TMM stack-of-stacks using newly found analytical functions for T and A

A = 5
gam = 0.001
nb = 1.7

#n_list, d_list = tmm_h.generate_n_and_d(gam, A, nb, min_thickness=0.0001, plot_flag=True)
n_list, d_list = tmm_h.generate_n_and_d_new(gam, A, nb, delta=0.03, plot_flag=False)

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
#print(R_list_RL)

FOM_RL = np.trapz(T_list_RL, x=lambda_list)**2 / np.trapz(A_list_RL, x=lambda_list) / delta_lamb
FOM_LR = np.trapz(T_list_LR, x=lambda_list)**2 / np.trapz(A_list_LR, x=lambda_list) / delta_lamb
ASYM = np.trapz(A_list_LR, x=lambda_list) / np.trapz(A_list_RL, x=lambda_list)
FOM_bulk = np.trapz(trans_bulk, x=lambda_list)**2 / np.trapz(emiss_bulk, x=lambda_list) / delta_lamb

# %% ############################################################################################

# plotting using Will's plot modules

xlabel = 'Wavelength ($\mu$m)'; ylabel = 'Fraction of Power'
#title = f'Transmittance (A={A}, x$_0$={gam}$\mu$m)'
title = 'Reflectance or Absorbance'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

#plot(fig,ax,lambda_list,T_list_RL,label=r'T',color=colors.blue,auto_scale=True)
#plot(fig,ax,lambda_list,T_list_LR, '--', label=r'T$_{LR}$',color=colors.light_blue,auto_scale=True)
#plot(fig,ax,lambda_list,trans_bulk, '--', label=r'T$_{bulk}$',color=colors.light_blue,auto_scale=True)

plot(fig,ax,lambda_list,A_list_RL,'<-', markersize=8, markevery=15, label=r'A$_{{RL}}$',color=colors.red,auto_scale=True)
plot(fig,ax,lambda_list,A_list_LR, '>-', markersize=8, markevery=15, label=r'A$_{LR}$',color=colors.red,auto_scale=True)
#plot(fig,ax,lambda_list,emiss_bulk, '--', label=r'A$_{bulk}$',color=colors.light_red,auto_scale=True)

plot(fig,ax,lambda_list,R_list_RL, '<-', markersize=8, markevery=15, label=r'R$_{RL}$',color=colors.green,auto_scale=True)
plot(fig,ax,lambda_list,R_list_LR, '>-', markersize=8, markevery=15, label=r'R$_{LR}$',color=colors.green,auto_scale=True)

text_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#ax.text(0.5, 0.75, f"T$_{{avg}}$ = {round(T_avg, 3)}\nA$_{{LR}}$/A$_{{RL}}$ = {round(ASYM, 3)}\nFOM = {round(FOM, 3)}", size='x-large', bbox=text_box, ha='left', va='top', transform=ax.transAxes)

legend(fig,ax,auto_scale=True)
#%% ############################################################################################

N = 50
T2, R2_LR, R2_RL, A2_LR, A2_RL, emiss_bulk, trans_bulk = (np.empty((len(lambda_list), N), dtype=float) for i in range(7))
FOM_RL, FOM_LR, ASYM, FOM_bulk, FOM_enh = (np.empty(N, dtype=float) for i in range(5))

for n in range(1,N+1):
    print(f'num of stacks is: {n}')
    j = n-1
    T2[:,j] = T_list_LR**n
    geom_series = np.zeros_like(lambda_list)
    for l in range(1,n):
        #print(f"entered! l = {l}, n = {n}")
        geom_series = geom_series + T_list_LR**l
    
    #A2_LR[:,j] = A_list_LR*geom_series
    A2_RL[:,j] = A_list_RL + (A_list_RL + A_list_LR*R_list_RL)*geom_series

    trans_bulk[:,j] = np.exp(-4*np.pi*n*losses_total/lambda_list)
    emiss_bulk[:,j] = 1 - trans_bulk[:,j]
    FOM_RL[j] = np.trapz(T2[:,j], x=lambda_list)**2 / np.trapz(A2_RL[:,j], x=lambda_list) / delta_lamb
    #FOM_LR[j] = np.trapz(T2[:,j], x=lambda_list)**2 / np.trapz(A2_LR[:,j], x=lambda_list) / delta_lamb
    #ASYM[j] = np.trapz(A2_LR[:,j], x=lambda_list) / np.trapz(A2_RL[:,j], x=lambda_list)
    FOM_bulk[j] = np.trapz(trans_bulk[:,j], x=lambda_list)**2 / np.trapz(emiss_bulk[:,j], x=lambda_list) / delta_lamb
    FOM_enh[j] = FOM_RL[j] / FOM_bulk[j]

print(f'FOM for 1 KK: {FOM_RL[0]}')
print(f'FOM for 1 bulk: {FOM_bulk[0]}')
print(f'FOM enhancement for 1: {FOM_RL[0]/FOM_bulk[0]}')
print(f'FOM for {N} KK: {FOM_RL[N-1]}')
print(f'FOM for {N} bulk: {FOM_bulk[N-1]}')
print(f'FOM enhancement for {N}: {FOM_RL[N-1]/FOM_bulk[N-1]}')
# %%
print(FOM_enh)
xlabel = 'Number of Stacks'; ylabel = 'FOM enhancement'
title = f'A = {A}, x$_0$ = {gam}'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(1,N),figsize=(5,4),auto_scale=True)

plot(fig,ax,range(1,N+1),FOM_enh, color=colors.blue, label='KK',auto_scale=True)
#plot(fig,ax,range(N),np.log(FOM_bulk), '--', color=colors.red, label='bulk',auto_scale=True)

legend(fig,ax,auto_scale=True)
# %%
xlabel = 'Wavelength'; ylabel = ''
title = f'A = {A}, x$_0$ = {gam}'
fig,ax = plot_setup(xlabel,ylabel,title=title,xlim=(lambda_list[0],lambda_list[-1]),figsize=(5,4),auto_scale=True)

plot(fig,ax,lambda_list,T2[:,N-1], color=colors.blue, label='T',auto_scale=True)
plot(fig,ax,lambda_list,A2_RL[:,N-1], color=colors.green, label='A',auto_scale=True)

#plot(fig,ax,range(N),np.log(FOM_bulk), '--', color=colors.red, label='bulk',auto_scale=True)

legend(fig,ax,auto_scale=True)
# %%
