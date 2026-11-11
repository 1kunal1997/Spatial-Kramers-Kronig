import numpy as np, sys
sys.path.insert(0, '.')
import cfreq_lib as cf
# 1. vacuum slab: r=0, tau=1
y, chi, d = cf.discretize(lambda y: 0*y, -1, 1, 50)
o = cf.rt(1+chi, d, [cf.omega_c(0.0), cf.omega_c(0.1)])
print('vacuum', np.abs(o['rL']), np.abs(o['tauL']-1))
# 2. homogeneous slab vs Fresnel/Airy at real w
n=1.5; L=0.37
y, chi, d = cf.discretize(lambda y: 0*y+(n**2-1), 0, L, 7)
w=cf.TWO_PI; k=w
r12=(1-n)/(1+n); ph=np.exp(2j*n*k*L)
r_airy=(r12 - r12*ph)/(1-r12**2*ph)
o=cf.rt(1+chi,d,[w]); print('airy', o['rL'], r_airy)
# 3. Lorentz design reproduces target at wc, and is passive at real w
wc=cf.omega_c(0.05)
chi=np.array([0.3+0.2j, -0.4-0.5j, 0.1-0.05j, 1e-6+0j])
p=cf.lorentz_design(chi, wc, 0.9*cf.kappa_max(chi,wc))
print('design ok', p['ok'], np.abs(cf.eps_lorentz([wc],p)[0]-1-chi).max())
ww=np.linspace(0.01,40,20000); print('min Im eps real w', cf.eps_lorentz(ww,p).imag.min())
# 4. sKK Lorentzian (passive, A<0) at real w: R_left << R_right
Y=40
y, chi, d = cf.discretize(lambda y: cf.lorentzian_profile(y,-0.3,0.1), -Y, Y, 40000)
o=cf.rt(1+chi,d,[cf.TWO_PI]); print('lorentz real', abs(o['rL'])**2, abs(o['rR'])**2, abs(o['tauL']))
