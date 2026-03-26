# %%

import tmm_helper as tmm_h
import numpy as np

# %% ##############################################################################################

#A = 2*1.89
A = 2*1.8
gam = 0.01
nb = 1

n_list, d_list = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, delta=0.1, plot_flag=True, zoomed=True)

max_index = np.argmax(np.array(n_list).real)

print(f"max index is: {max_index} and n_list here is: {n_list[max_index]}")

#for i, n in enumerate(n_list):
#    print(n)
max_index = 18
n_coating = n_list[0:max_index+1]
#n_coating = [num.real for num in n_coating]
d_coating = d_list[0:max_index+1]

k_window = 1e-5
n_window = n_coating[-1].real
c_total = []
for _ in range(len(n_coating)):
    c_total.append('c')
n_total = n_coating
n_total.append(n_window + 1j*k_window)
d_total = d_coating
d_total.append(5000)
c_total.append('i')
#for i, n in enumerate(n_total):
#   print(f'n is: {n}, d is: {d_total[i]}, c is: {c_total[i]}')

# %% #############################################################################################

lambda_list = np.linspace(2,5,100)
delta_lamb = lambda_list[-1] - lambda_list[0]
#losses_total = np.sum(d_coating * np.imag(n_coating))
#trans_bulk = np.exp(-4*np.pi*losses_total/lambda_list)
#emiss_bulk = 1 - trans_bulk
#print(losses_total)

# add semi-infinite layers
d_total.append(np.inf)
d_total.insert(0, np.inf)
#n_coating.append(n_coating[-1])  
n_total.append(1)     
n_total.insert(0, 1)
#n_total.insert(0, n_window)
c_total.append('i')
c_total.insert(0, 'i')

for i, n in enumerate(n_total):
   print(f'n is: {n}, d is: {d_total[i]}, c is: {c_total[i]}')
n_total_reversed = n_total[::-1]
d_total_reversed = d_total[::-1]
c_total_reversed = c_total[::-1]

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_wavelength_inc(n_total, d_total, c_total, lambda_list)
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_wavelength_inc(n_total_reversed, d_total_reversed, c_total_reversed, lambda_list)

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
    #"A_LR": A_list_LR,
    #"R_RL": R_list_RL,
    "R_LR": R_list_LR,
}

tmm_h.plot_tra_curves(
    lambda_list,
    data=data,
    title='AR coating (sKK with no losses)'
    #title='Truncated sKK Coating'
)

tmm_h.show_textbox(text=f'n$_b$={nb}\nA={A}\nx$_0$={gam}')
# %%

noise = np.trapezoid(A_list_RL, x=lambda_list) / delta_lamb
signal = np.trapezoid(T_list_LR, x=lambda_list) / delta_lamb
FOM = signal / noise
refl = np.trapezoid(R_list_LR, x=lambda_list) / delta_lamb

print(f'average T is: {signal}')
print(f'average E is: {noise}')
print(f'average FOM is: {FOM}')
print(f'average refl is: {refl}')

# %% ##############################################################################################

#A = 2*1.89
A = 2*1.8
gam = 0.01
nb = 1

n_list, d_list = tmm_h.generate_n_and_d_v6_symmetry(gam, A, nb, delta=0.1, plot_flag=False, zoomed=True)

max_index = np.argmax(np.array(n_list).real)

print(f"max index is: {max_index} and n_list here is: {n_list[max_index]}")

#for i, n in enumerate(n_list):
#    print(n)
max_index = 18
n_coating = n_list[0:max_index+1]
n_coating = [num.real for num in n_coating]
d_coating = d_list[0:max_index+1]

k_window = 1e-5
n_window = n_coating[-1].real
c_total = []
for _ in range(len(n_coating)):
    c_total.append('c')
n_total = n_coating
n_total.append(n_window + 1j*k_window)
d_total = d_coating
d_total.append(5000)
c_total.append('i')
#for i, n in enumerate(n_total):
#   print(f'n is: {n}, d is: {d_total[i]}, c is: {c_total[i]}')

# %% #############################################################################################

angle_list = np.linspace(0,80,200)
delta_angle = angle_list[-1] - angle_list[0]
degrees = np.pi/180
#losses_total = np.sum(d_coating * np.imag(n_coating))
#trans_bulk = np.exp(-4*np.pi*losses_total/lambda_list)
#emiss_bulk = 1 - trans_bulk
#print(losses_total)

# add semi-infinite layers
d_total.append(np.inf)
d_total.insert(0, np.inf)
#n_coating.append(n_coating[-1])  
n_total.append(1)     
n_total.insert(0, 1)
#n_total.insert(0, n_window)
c_total.append('i')
c_total.insert(0, 'i')

for i, n in enumerate(n_total):
   print(f'n is: {n}, d is: {d_total[i]}, c is: {c_total[i]}')
n_total_reversed = n_total[::-1]
d_total_reversed = d_total[::-1]
c_total_reversed = c_total[::-1]

T_list_LR, R_list_LR, A_list_LR = tmm_h.TRA_angle_inc(n_total, d_total, c_total, angle_list*degrees, pol='p')
T_list_RL, R_list_RL, A_list_RL = tmm_h.TRA_angle_inc(n_total_reversed, d_total_reversed, c_total_reversed, angle_list*degrees, pol='p')

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
    #"A_LR": A_list_LR,
    #"R_RL": R_list_RL,
    #"R_LR": R_list_LR,
}

tmm_h.plot_tra_curves(
    angle_list,
    data=data,
    xlabel='Angle (degrees)',
    title='AR coating (p-pol)'
    #title='Truncated sKK Coating (p-pol)'
)

tmm_h.show_textbox(text=f'n$_b$={nb}\nA={A}\nx$_0$={gam}')
# %%

noise = np.trapezoid(A_list_RL, x=angle_list) / delta_angle
signal = np.trapezoid(T_list_LR, x=angle_list) / delta_angle
FOM = signal / noise
refl = np.trapezoid(R_list_LR, x=angle_list) / delta_angle

print(f'average T is: {signal}')
print(f'average E is: {noise}')
print(f'average FOM is: {FOM}')
print(f'average refl is: {refl}')

# %%
