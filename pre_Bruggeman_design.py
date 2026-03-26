# %%

import tmm_helper as tmm_h
import numpy as np
import tmm

# %% ##############################################################################################

A = 5
gam = 0.04
nb = 2.3

n_list, d_list = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, delta=0.2, plot_flag=True, zoomed=False)

max_index = np.argmax(np.array(n_list).real)
min_index = np.argmin(np.array(n_list).real)

print(f"min index is: {min_index} and n_list here is: {n_list[min_index]}")
print(f"max index is: {max_index} and n_list here is: {n_list[max_index]}")

#for i, n in enumerate(n_list):
#    print(n)

n_coating = n_list[max_index:min_index+1]
n_coating = [num.real for num in n_coating]
d_coating = d_list[max_index:min_index+1]
for i, n in enumerate(n_coating):
    print(n)

# %% #############################################################################################

lambda_list = np.linspace(2,5,100)
delta_lamb = lambda_list[-1] - lambda_list[0]
#losses_total = np.sum(d_coating * np.imag(n_coating))
#trans_bulk = np.exp(-4*np.pi*losses_total/lambda_list)
#emiss_bulk = 1 - trans_bulk
#print(losses_total)

# add semi-infinite air layers
d_coating.append(np.inf)
d_coating.insert(0, np.inf)
#n_coating.append(n_coating[-1].real)  
n_coating.append(1)     
n_coating.insert(0, n_coating[0].real)

for i, n in enumerate(n_coating):
    print(n)

n_coating_reversed = n_coating[::-1]
d_coating_reversed = d_coating[::-1]

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_wavelength(n_coating, d_coating, lambda_list)
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_wavelength(n_coating_reversed, d_coating_reversed, lambda_list)

# %% #############################################################################################

'''
data = {
    "T": T_list_LR,      # or T_list_RL, doesn't matter anymore
    "A_RL": A_list_RL,
    "A_LR": A_list_LR,
    "R_RL": R_list_RL,
    "R_LR": R_list_LR,
    "T_bulk": trans_bulk,
    "A_bulk": emiss_bulk
}
'''
data = {
    "T": T_list_LR,      # or T_list_RL, doesn't matter anymore
    #"A_RL": A_list_RL,
    "A_LR": A_list_LR,
    #"R_RL": R_list_RL,
    "R_LR": R_list_LR,
}

tmm_h.plot_tra_curves(
    lambda_list,
    data=data,
    #title='AR coating (sKK with no losses)'
    title='Truncated sKK Coating'
)


tmm_h.show_textbox(text=f'n$_b$={nb}\nA={A}\nx$_0$={gam}')
# %% ##############################################################################################

R_noise = np.trapezoid(R_list_LR, x=lambda_list) / delta_lamb
E_noise = np.trapezoid(A_list_LR, x=lambda_list) / delta_lamb
noise = R_noise + E_noise

print(f'average reflectance is: {R_noise}')
print(f'average emittance is: {E_noise}')
print(f'total noise is: {noise}')

# %% ##############################################################################################

A = 5
gam = 0.04
nb = 2.3

n_coating, d_coating = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, delta=0.2, plot_flag=True, zoomed=False)

n_window = n_coating[0].real
c_total = []
for _ in range(len(n_coating)):
    c_total.append('c')
n_total = n_coating
n_total.insert(0, n_window)
d_total = d_coating
d_total.insert(0, 4000)
c_total.insert(0, 'i')

# %% #############################################################################################

angle_list = np.linspace(0,80,200)
delta_angle = angle_list[-1] - angle_list[0]
degrees = np.pi/180
lamb = 3
pol = 'p'

# add semi-infinite air layers
d_coating.append(np.inf)
d_coating.insert(0, np.inf)
n_coating.append(1)  
#n_coating.append(1)     
n_coating.insert(0, 1)

c_total.append('i')
c_total.insert(0, 'i')

for i, n in enumerate(n_total):
   print(f'n is: {n}, d is: {d_total[i]}, c is: {c_total[i]}')

n_total_reversed = n_total[::-1]
d_total_reversed = d_total[::-1]
c_total_reversed = c_total[::-1]

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle_inc(n_total, d_total, c_total, angle_list*degrees, pol=pol)
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_angle_inc(n_total_reversed, d_total_reversed, c_total_reversed, angle_list*degrees, pol=pol)


# %% #############################################################################################

'''
data = {
    "T": T_list_LR,      # or T_list_RL, doesn't matter anymore
    "A_RL": A_list_RL,
    "A_LR": A_list_LR,
    "R_RL": R_list_RL,
    "R_LR": R_list_LR,
    "T_bulk": trans_bulk,
    "A_bulk": emiss_bulk
}
'''
data = {
    "T": T_list_LR,      # or T_list_RL, doesn't matter anymore
    "A_RL": A_list_RL,
    "A_LR": A_list_LR,
    "R_RL": R_list_RL,
    "R_LR": R_list_LR,
}

tmm_h.plot_tra_curves(
    angle_list,
    data=data,
    #title='AR coating (p-pol)',
    title='sKK coating (p-pol)',
    xlabel='Angle (degrees)'
)


tmm_h.show_textbox(text=f'n$_b$={nb}\nA={A}\nx$_0$={gam}')
# %%

R_noise_int = np.trapezoid(R_noise, x=angle_list) / delta_angle
E_noise_int = np.trapezoid(A_list_LR, x=angle_list) / delta_angle
noise = R_noise_int + E_noise_int

print(f'average reflectance is: {R_noise_int}')
print(f'average emittance is: {E_noise_int}')
print(f'total noise is: {noise}')

# %%
