"""Exponentially-tapered Lorentzian eps': what does its Hilbert partner look like?

eps'_mod(y) = eps_b + [-s*A*gam*u/D] * sech(u/ell)      u=y-y0, D=u^2+gam^2
eps''       = A*gam^2/D   (UNCHANGED -> passive by construction)

Then compute the HT partner of eps'_mod two ways (direct HT, DiHiTI) and see
whether either is passive.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tmm_helper as tmm_h
from scipy.signal import hilbert
HERE = os.path.dirname(os.path.abspath(__file__))

lam=0.8; eb=1.7**2; A=4.0; gam=lam/5

# big padded window so the 1/u tail is resolved for the HT
W = 400*lam; N = 2**21
u = np.linspace(-W/2, W/2, N)
D = u**2 + gam**2
dre_lor = -A*gam*u/D          # Lorentzian real perturbation (s=+1 sign here)
im_lor  =  A*gam**2/D         # its exact KK partner

# --- 1. fix the HT sign convention using the exact Lorentzian pair ----------
h = np.imag(hilbert(dre_lor))
sgn = +1 if np.corrcoef(h, im_lor)[0,1] > 0 else -1
err = np.max(np.abs(sgn*h - im_lor))/im_lor.max()
print(f'HT convention: eps_im = {sgn:+d}*Im(hilbert(eps_re)); check on exact Lorentzian '
      f'-> max rel err {err:.2e}')
print(f'  Int eps_im dy (Lorentzian) = {np.trapezoid(im_lor,u):.4f}  (theory pi*A*gam = {np.pi*A*gam:.4f})')
print(f'  1/u tail coeff of eps_re   = {A*gam:.4f}   = (1/pi)*Int eps_im dy = {np.trapezoid(im_lor,u)/np.pi:.4f}\n')

# --- 2. taper the real part, keep eps'' ------------------------------------
out={}
for ell_over_lam in [0.5, 1.0, 2.0]:
    ell = ell_over_lam*lam
    taper = 1.0/np.cosh(u/ell)
    dre_t = dre_lor*taper
    ht_direct = sgn*np.imag(hilbert(dre_t))
    # DiHiTI on the same (restricted to a finite domain like the model uses)
    L = 8*lam
    m = (u >= -L/2) & (u <= L/2)
    ht_dihiti = tmm_h.ht_derivative(u[m], eb + dre_t[m])
    I_direct = np.trapezoid(ht_direct, u)
    I_dih    = np.trapezoid(ht_dihiti, u[m])
    print(f'ell = {ell_over_lam}*lam:')
    print(f'  tapered eps_re at |u|=L/2 : {abs(dre_t[m][0]):.3e}   (untapered: {abs(dre_lor[m][0]):.3e})')
    print(f'  eps_im (unchanged Lorentzian): min={im_lor[m].min():+.4f}  Int={np.trapezoid(im_lor[m],u[m]):.4f}')
    print(f'  HT direct  : min={ht_direct.min():+.4f}  max={ht_direct.max():+.4f}  Int={I_direct:+.4e}  PASSIVE={ht_direct.min()>-1e-6}')
    print(f'  HT DiHiTI  : min={ht_dihiti.min():+.4f}  max={ht_dihiti.max():+.4f}  Int={I_dih:+.4e}  PASSIVE={ht_dihiti.min()>-1e-6}')
    print()
    out[ell_over_lam]=(u[m], dre_t[m], im_lor[m], ht_direct[m], ht_dihiti)

np.savez(os.path.join(HERE,'taper_ht.npz'),
         **{f'ell{k}_{n}':v for k,(uu,a,b,c,d) in out.items()
            for n,v in [('u',uu),('dre',a),('im_lor',b),('ht_direct',c),('ht_dihiti',d)]})
print('L1 test: Int|eps_re-eps_b| dy    Lorentzian(diverges as log) vs tapered(finite)')
for ell_over_lam in [0.5,1.0,2.0]:
    t = 1.0/np.cosh(u/(ell_over_lam*lam))
    print(f'  ell={ell_over_lam}lam : tapered Int|.| = {np.trapezoid(np.abs(dre_lor*t),u):.4f}'
          f'   untapered = {np.trapezoid(np.abs(dre_lor),u):.4f}')
