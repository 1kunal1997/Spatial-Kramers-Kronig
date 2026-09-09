import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
t=np.load(os.path.join(HERE,'skk_nb17_thickness.npz'))
pr=np.load(os.path.join(HERE,'profile_nb17.npz'))
nb=1.7; th_c=np.degrees(np.arcsin(1/nb))

fig,(a0,a1)=plt.subplots(1,2,figsize=(11.5,4.6))
a0.plot(pr['y']*1e6,pr['eps_re'],color='#1f77b4',lw=2,label=r"$\epsilon'(y)$")
a0.plot(pr['y']*1e6,pr['eps_im'],color='#8B0000',lw=2,label=r"$\epsilon''(y)$")
a0.axhline(nb**2,color='#1f77b4',ls=':',lw=1); a0.axhline(1,color='0.6',ls=':',lw=1)
a0.set_xlabel(r'$y$ ($\mu$m)'); a0.set_ylabel(r'$\epsilon$')
a0.text(1.55,2.95,'entry ($n_b$)',ha='right',fontsize=8,color='0.35')
a0.text(0.05,1.05,'air / PEC',ha='left',fontsize=8,color='0.35')
a0.set_title(r'logistic sKK, $n_b=1.7\rightarrow$ air ($L=2\lambda$, $w=L/40$)',fontsize=10)
a0.legend(fontsize=9); a0.grid(alpha=0.3)

cols=plt.cm.viridis(np.linspace(0.1,0.85,5))
for c,m in zip(cols,[2,4,6,8,12]):
    d=t[f'sKK_L{m}']; a1.semilogy(d[0],d[1],'-',color=c,lw=1.6,label=rf'sKK, $L={m}\lambda$')
a1.semilogy(t['PML_pf1'][0],t['PML_pf1'][1],'k-',lw=2.2,label=r'PML pf=1 ($L=2\lambda$)')
a1.semilogy(t['PML_pf4'][0],t['PML_pf4'][1],'k--',lw=2.2,label=r'PML pf=4 ($L=2\lambda$)')
a1.axvline(th_c,color='r',ls=':',lw=1.2)
a1.text(th_c+1,3e-11,r'$\theta_c=36°$',color='r',fontsize=8,rotation=90,va='bottom')
a1.set_xlabel(r'$\theta$ in substrate (deg)'); a1.set_ylabel(r'Reflectance $|S_{11}|^2$')
a1.set_xlim(0,89); a1.set_ylim(1e-11,2); a1.grid(alpha=0.3)
a1.legend(fontsize=7.5,loc='lower right',ncol=2)
a1.set_title(r'sKK terminator vs PML, $n_b=1.7$ background',fontsize=10)
fig.tight_layout(); p=os.path.join(HERE,'R_vs_angle_nb17.png'); fig.savefig(p,dpi=200)
print('saved',p)

print('\nangles where sKK (L=12lam) beats PML pf=1:')
d=t['sKK_L12']; P=t['PML_pf1']
win=d[0][(d[1]<P[1])]
print(f'  {win.min():.0f}-{win.max():.0f} deg' if len(win) else '  none')
r=P[1]/d[1]
i=int(np.argmax(r)); print(f'  best advantage {r[i]:.0f}x at {d[0][i]:.0f} deg')
