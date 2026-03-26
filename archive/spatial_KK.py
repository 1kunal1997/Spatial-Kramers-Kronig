from numpy import linspace,floor,abs,insert,append,diff,sqrt,inf,cumsum,real,imag,pi,array,exp,cos,trapz
import matplotlib.pyplot as plt
import tmm
deg = pi/180
#==============================================================================================
# Inputs
#==============================================================================================
A       = 1                     # Lorentzian amplitude
x0      = 0.01                  # Lorentzian width (in microns)
th      = 00 * deg              # Incident light angle (normal = 0)
pol     = 's'                   # Polarization 
nb      = 1.7                   # Background refractive index (approximating sapphire)
nw      = 20                    # Number of wavelengths in the calculation

del_e   = A/25                  # Max e-step size in discrete Lorentzian approximation
del_x   = 15*x0                 # Max x-step size in discrete Lorentzian approximation
dx      = x0/100                # Step size in 'continuous' Lorentzian
xmin    = -x0 * 200             # Limits of Lorentzian
xmax    = - xmin

wls = linspace(2,5,nw)
#==============================================================================================
# Functions
#==============================================================================================
def Lor(x,A,x0,eb):
    return eb - A * x0 / (x + 1j * x0) 


def calc_TRA(d_list, n_list, wls, th, pol):
    coh_length = 5 #Maximum coherence length, in microns
    nw      = len(wls)
    c_list = ['c' if d<coh_length else 'i' for d in d_list]
    R,T = [] , []
    for kw in range(0,nw):
        tmp = tmm.tmm_core.inc_tmm(pol, n_list, d_list, c_list, th, wls[kw])
        R.append(tmp['R']); T.append(tmp['T'])
        

    R,T = array(R) , array(T); A = 1 - R - T
    return T , R , A


#==============================================================================================
# Interpolation
#==============================================================================================
nx      = 1 + int(floor((xmax - xmin) / dx))
xx      = linspace(xmin , xmax , nx )           # Approximately continuous x-axis
ee      = Lor(xx,A,x0,nb**2)                    # Smooth Lorentzian curve
count   = 0                                     # Counter for number of points in descrete approximation
xq      = [xx[0]]                               
eq      = [nb**2]
for k in range(0,nx):
    if abs((ee[k]) - (eq[count])) > del_e or abs((xx[k]) - (xq[count])) > del_x:
        xq.append(xx[k]); eq.append(ee[k])
        count = count + 1


xq = append(xq,xmax)
eq = append(eq,nb**2)
x_lay   = (xq[1:] + xq[0:-1])/2                                     # Form N-1 vector of midpoints
d_list  = diff(x_lay)
n_list  = sqrt(eq[1:-1])
x_lay = insert(x_lay,0,xq[0]); x_lay = append(x_lay,xq[-1])
d_list = insert(d_list,0,inf); d_list = append(d_list,inf)          # Append super- & sub-strate
n_list = insert(n_list,0,nb); n_list = append(n_list,nb)
#==============================================================================================
# Calculations
#==============================================================================================
T , Rf , Ef = calc_TRA(d_list, n_list, wls, th, pol)
T , Rb , Eb = calc_TRA(d_list[::-1], n_list[::-1], wls, th, pol)
kd          = sum(d_list [1:-1] * imag(n_list[1:-1]))
Tbu         = exp(-4 * pi * kd / wls / cos(th))
Ebu         = 1 - Tbu
FoM         = trapz(T,wls)**2 / trapz(Eb,wls) / trapz(wls/wls , wls)
FoM_bu      = trapz(Tbu,wls)**2 / trapz(Ebu,wls) / trapz(wls/wls , wls)
#==============================================================================================
# Plotting
#==============================================================================================
plt.plot(xx/x0 , real(sqrt(ee)))
plt.step(xq/x0 , real(sqrt(eq)) , where='mid')
plt.plot((x_lay[1:] + x_lay[0:-1])/2/x0 , real(n_list) , '*')
plt.title('%d layer approximation'%len(xq))
plt.xlabel('Position (x0)')
#plt.xlim([xmin/2/x0,xmax/2/x0])
plt.ylabel('Real refractive index')

plt.figure(2)
plt.plot(xx/x0 , imag(sqrt(ee)))
plt.step(xq/x0 , imag(sqrt(eq)) , where='mid')
plt.plot((x_lay[1:] + x_lay[0:-1])/2/x0 , imag(n_list) , '*')
plt.title('%d layer approximation'%len(xq))
plt.xlabel('Position (x0)')
#plt.xlim([-5,5])
plt.ylabel('Imag refractive index')

plt.figure(3)
plt.plot(wls , T)
plt.plot(wls , Ef)
plt.plot(wls , Eb)
plt.plot(wls , Tbu,'--')
plt.plot(wls , Ebu,'--')
plt.legend(['Transmittance' , 'Forward emittance' , 'Backward emittance','Bulk transmittance' , 'Bulk emittance'])
plt.xlabel('Wavelength (microns)')
plt.ylabel('Transmittance or emittance')
plt.title('FoM = %g & Bulk FoM = %g ---> %gx'%(FoM,FoM_bu,FoM/FoM_bu))

plt.figure(4)
plt.plot(wls , Rf)
plt.plot(wls , Rb)
plt.legend([ 'Forward' , 'Backward'])
plt.xlabel('Wavelength (microns)')
plt.ylabel('Reflectance')

plt.show()