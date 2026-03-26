import numpy as np
import matplotlib.pyplot as plt
import tmm
deg = np.pi/180
#==============================================================================================
# Inputs
#==============================================================================================
A       = 1.0                   # Lorentzian amplitude
x0      = 0.01                  # Lorentzian width (in microns)
th      = 0 * deg               # Incident light angle (normal = 0)
pol     = 'p'                   # Polarization 
t_KK    = 100 * x0              # Thickness of KK layer
t_bulk  = 1                     # Thickness of bulk/incoherent layers
n_lay   = 20                     # Number of [KK + bulk] layers in the stack of stacks
nb      = 1.7                   # Background refractive index (approximating sapphire)
nw      = 100                   # Number of wavelengths in the calculation

del_e   = A/25                  # Max e-step size in Lorentzian discretization
del_x   = 10*x0                 # Max x-step size in Lorentzian discretization
dx      = x0/250                # Step size in 'continuous' Lorentzian
xmin    = -x0 * 200             # Limits of Lorentzian

wls = np.linspace(2,5,nw)
#==============================================================================================
# Functions
#==============================================================================================
def Lor(x,A,x0,eb):
    return eb - A * x0 / (x + 1j * x0) 


def calc_TRA(d_list, n_list, c_list, wls, th, pol):
    nw      = len(wls)
    R,T = [] , []
    for kw in range(0,nw):
        tmp = tmm.tmm_core.inc_tmm(pol, n_list, d_list, c_list, th, wls[kw])
        R.append(tmp['R']); T.append(tmp['T'])
        

    R,T = np.array(R) , np.array(T); A = 1 - R - T
    return T , R , A


def calc_FoM(wls,T,E):
    return np.trapz (T,wls)**2 / np.trapz(E,wls) / np.trapz(wls/wls , wls)


#==============================================================================================
# Interpolation
#==============================================================================================
xmax    = - xmin
nx      = 1 + int(np.floor((xmax - xmin) / dx))
xx      = np.linspace(xmin , xmax , nx )
ee      = Lor(xx,A,x0,nb**2)                            
count   = 0
xq      = [xx[0]]
eq      = [nb**2]
for k in range(0,nx):
    if abs((ee[k]) - (eq[count])) > del_e or abs((xx[k]) - (xq[count])) > del_x:
        xq.append(xx[k]); eq.append(ee[k])
        count = count + 1


xq = np.append(xq,xmax)
eq = np.append(eq,nb**2)
#==============================================================================================
# Generate stacks
#==============================================================================================
x_lay   = (xq[1:] + xq[0:-1])/2                     # Interpolation mid-points
d_KK_l  = np.diff(x_lay)                            # List of KK layer thicknesses
n_KK_l  = np.sqrt(eq[1:-1])                         # List of KK refractive indices
c_KK_l  = ['c'] * len(d_KK_l)
kd_KK   = sum(d_KK_l * np.imag(n_KK_l))             # Total losses in KK layer
k_avg   = kd_KK / t_KK                              # Avg k based on total KK losses & thickness

n_bu_l = np.array([nb + 1j * k_avg])                #Bulk/incoherent layer refractive index
d_bu_l = np.array([t_bulk])                         #Bulk/incoherent layer thickness
c_bu_l = ['i']

n_list = np.array([])
d_list = np.array([])
c_list = []

for kk in range(0 , n_lay):                         #Make stack of stacks
    n_list = np.concatenate((n_list , n_KK_l , n_bu_l))
    d_list = np.concatenate((d_list , d_KK_l , d_bu_l))
    c_list = c_list + c_KK_l + c_bu_l


d_list = np.insert(d_list,0,np.inf); d_list = np.append(d_list,np.inf) # Add semi-inf layers (super/sub-strates)
n_list = np.insert(n_list,0,nb);     n_list = np.append(n_list,nb)
c_list = ['i'] + c_list + ['i']
#==============================================================================================
# Calculations
#==============================================================================================
T , Rf , Ef = calc_TRA(d_list, n_list, c_list, wls, th, pol)
T , Rb , Eb = calc_TRA(d_list[::-1], n_list[::-1], c_list[::-1], wls, th, pol)
kd          = k_avg * (t_KK + t_bulk) * n_lay
Tbu         = np.exp(-4 * np.pi * kd / wls / np.cos(th))
Ebu         = 1 - Tbu
FoM         = calc_FoM(wls,T,Eb)
FoM_bu      = calc_FoM(wls,Tbu,Ebu)
#==============================================================================================
# Plotting
#==============================================================================================
plt.plot(wls , T)
plt.plot(wls , Ef)
plt.plot(wls , Eb)
plt.plot(wls , Tbu,'--')
plt.plot(wls , Ebu,'--')
plt.legend(['Transmittance' , 'Forward emittance' , 'Backward emittance','Bulk transmittance' , 'Bulk emittance'])
plt.xlabel('Wavelength (microns)')
plt.ylabel('Transmittance or emittance')
plt.title('FoM = %g & Bulk FoM = %g ---> %gx'%(FoM,FoM_bu,FoM/FoM_bu))


plt.show()