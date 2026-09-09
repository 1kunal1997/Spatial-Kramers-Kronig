import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
st=np.load(os.path.join(HERE,'skk_stacked_v1.npz'))
th_=np.load(os.path.join(HERE,'skk_nb17_thickness.npz'))
nb=1.7; th_c=np.degrees(np.arcsin(1/nb))

fig,ax=plt.subplots(figsize=(7.2,5.2))
cols=plt.cm.plasma(np.linspace(0.12,0.72,3))
for c,m in zip(cols,[2,4,6]):
    d=st[f'stacked_L{m}']
    ax.semilogy(d[0],d[1],'-',color=c,lw=2.0,label=rf'stacked: sKK $L={m}\lambda$ + PML $\lambda$')
for ls,m in [((0,(4,2)),2),((0,(1,1.5)),12)]:
    d=th_[f'sKK_L{m}']
    ax.semilogy(d[0],d[1],ls=ls,color='#8B0000',lw=1.5,label=rf'sKK alone, PEC-backed, $L={m}\lambda$')
ax.semilogy(th_['PML_pf1'][0],th_['PML_pf1'][1],'k-',lw=2.2,label=r'PML alone, pf=1 ($2\lambda$)')
ax.semilogy(th_['PML_pf4'][0],th_['PML_pf4'][1],'k--',lw=1.8,label=r'PML alone, pf=4 ($2\lambda$)')
ax.axvline(th_c,color='r',ls=':',lw=1.2)
ax.text(th_c+1,2e-11,r'$\theta_c=36°$',color='r',fontsize=8,rotation=90,va='bottom')
ax.set_xlabel(r'$\theta$ in substrate (deg)'); ax.set_ylabel(r'Reflectance $|S_{11}|^2$')
ax.set_xlim(0,89); ax.set_ylim(1e-11,2); ax.grid(alpha=0.3)
ax.legend(fontsize=8,loc='lower right')
ax.set_title(r'Stacked terminator: substrate $\to$ sKK $\to$ air+PML $\to$ PEC   ($n_b=1.7$)',fontsize=10)
fig.tight_layout(); p=os.path.join(HERE,'R_vs_angle_stacked.png'); fig.savefig(p,dpi=200)
print('saved',p,'\n')

print(f'{"config":34s} {"R(0)":>10} {"R(20)":>10} {"R(35)":>10} {"R(60)":>10} {"R(89)":>8}')
def row(lab,d):
    i=[int(np.argmin(abs(d[0]-t))) for t in [0,20,35,60,89]]
    print(f'{lab:34s} '+' '.join(f'{d[1][k]:>10.2e}' for k in i[:4])+f' {d[1][i[4]]:>8.3f}')
for m in [2,4,6]: row(f'stacked sKK {m}lam + PML', st[f'stacked_L{m}'])
row('sKK alone PEC-backed 2lam', th_['sKK_L2'])
row('sKK alone PEC-backed 12lam', th_['sKK_L12'])
row('PML alone pf=1', th_['PML_pf1']); row('PML alone pf=4', th_['PML_pf4'])

d=st['stacked_L6']; P=th_['PML_pf1']
w=d[0][d[1]<P[1]]
print(f'\nstacked(6lam) beats PML pf=1 at: {w.min():.0f}-{w.max():.0f} deg'
      f'  (best {np.max(P[1]/d[1]):.0f}x at {d[0][int(np.argmax(P[1]/d[1]))]:.0f} deg)')
