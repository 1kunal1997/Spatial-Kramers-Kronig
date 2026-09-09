import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
OUT=HERE
log=np.load(os.path.join(HERE,'skk_logistic_v1.npz'))
lor=np.load(os.path.join(HERE,'skk_scan_v1.npz'))
pml=np.load(os.path.join(HERE,'pml_baseline_sweep.npz'))
prof=np.load(os.path.join(HERE,'logistic_profile_D3p6.npz'))

def fresnel_s(D, th):
    """R for an abrupt step eps 1 -> 1+D, s-pol"""
    c1=np.cos(np.radians(th)); s2=np.sin(np.radians(th))**2
    k2=np.sqrt((1+D)-s2+0j)
    return np.abs((c1-k2)/(c1+k2))**2

# ---- figure 1: logistic R vs angle ----
fig,ax=plt.subplots(figsize=(6.4,4.8))
ax.semilogy(pml['theta'],pml['R'],'k-',lw=2,label='PML (PMLfactor=1)')
cols=plt.cm.plasma(np.linspace(0.1,0.8,5))
for c,D in zip(cols,['1','2','3.6','5','8']):
    d=log[f'D{D}']; ax.semilogy(d[0],d[1],'-',color=c,lw=1.6,label=rf'logistic sKK, $D$={D}')
th=np.linspace(0,89,200)
ax.semilogy(th,fresnel_s(3.6,th),'--',color='0.5',lw=1.4,label=r'Fresnel step $1\to4.6$ (s-pol)')
ax.set_xlabel(r'$\theta$ (deg)'); ax.set_ylabel(r'Reflectance $|S_{11}|^2$')
ax.set_xlim(0,89); ax.set_ylim(1e-12,2); ax.grid(True,alpha=0.3)
ax.legend(fontsize=8,loc='lower right')
ax.set_title(r'Logistic (DiHiTI) sKK terminator vs PML  ($L=2\lambda$, $w=L/40$)',fontsize=10)
fig.tight_layout(); p1=os.path.join(OUT,'R_vs_angle_logistic_sKK.png'); fig.savefig(p1,dpi=200)

# ---- figure 2: all three terminators + the profile ----
fig,(a0,a1)=plt.subplots(1,2,figsize=(11,4.4))
a0.plot(prof['y']*1e6,prof['eps_re'],color='#1f77b4',lw=2,label=r"$\epsilon'(y)$")
a0.plot(prof['y']*1e6,prof['eps_im'],color='#8B0000',lw=2,label=r"$\epsilon''(y)$")
a0.axvline(1.6,color='0.6',ls=':',lw=1); a0.text(1.55,3.6,'entry',ha='right',fontsize=8,color='0.4')
a0.axvline(0.0,color='0.6',ls=':',lw=1); a0.text(0.05,3.6,'PEC',ha='left',fontsize=8,color='0.4')
a0.set_xlabel(r'$y$ ($\mu$m)'); a0.set_ylabel(r'$\epsilon$'); a0.legend(fontsize=9)
a0.set_title(r'logistic DiHiTI profile, $D=3.6$',fontsize=10); a0.grid(alpha=0.3)

a1.semilogy(pml['theta'],pml['R'],'k-',lw=2,label='PML')
a1.semilogy(lor['s-1_A2'][0],lor['s-1_A2'][1],color='#2ca02c',lw=1.8,label=r'Lorentzian sKK ($A$=2, best)')
a1.semilogy(log['D3.6'][0],log['D3.6'][1],color='#d62728',lw=1.8,label=r'logistic sKK ($D$=3.6, best)')
a1.semilogy(th,fresnel_s(3.6,th),'--',color='0.5',lw=1.4,label=r'Fresnel step $1\to4.6$')
a1.set_xlabel(r'$\theta$ (deg)'); a1.set_ylabel(r'Reflectance $|S_{11}|^2$')
a1.set_xlim(0,89); a1.set_ylim(1e-12,2); a1.grid(alpha=0.3); a1.legend(fontsize=8,loc='lower right')
a1.set_title('best of each terminator',fontsize=10)
fig.tight_layout(); p2=os.path.join(OUT,'R_vs_angle_terminator_comparison.png'); fig.savefig(p2,dpi=200)
print('saved:\n ',p1,'\n ',p2)

print('\ngrazing: is it just Fresnel from the net step?')
print(f'{"D":>5} {"R(89) sKK":>12} {"R(89) Fresnel":>14}')
for D in [1,2,3.6,5,8]:
    print(f'{D:>5} {log[f"D{D}"][1][-1]:>12.3f} {fresnel_s(D,89.0):>14.3f}')
