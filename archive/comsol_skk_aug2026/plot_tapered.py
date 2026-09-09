import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=os.path.dirname(os.path.abspath(__file__))
ht=np.load(os.path.join(H,'taper_ht.npz')); tp=np.load(os.path.join(H,'lorentzian_tapered.npz'))
ls_=np.load(os.path.join(H,'lorentzian_stacked.npz'),allow_pickle=True)
lam=0.8; eb=1.7**2

# ---- Figure 1: the profile and its Hilbert partners ----
fig,axs=plt.subplots(1,3,figsize=(15,4.4))
for ax,ell in zip(axs,[0.5,1.0,2.0]):
    u=ht[f'ell{ell}_u']/lam
    ax2=ax.twinx()
    ax.plot(u, eb+ht[f'ell{ell}_dre'], color='#1f77b4', lw=2, label=r"$\epsilon'$ tapered")
    ax.axhline(eb, color='#1f77b4', ls=':', lw=1)
    ax2.plot(u, ht[f'ell{ell}_im_lor'], color='#8B0000', lw=2, label=r"$\epsilon''$ Lorentzian (used)")
    ax2.plot(u, ht[f'ell{ell}_ht_direct'], color='#ff7f0e', lw=1.4, label=r"$\epsilon''=$HT$[\epsilon'_{tap}]$")
    ax2.plot(u, ht[f'ell{ell}_ht_dihiti'], color='#2ca02c', lw=1.4, ls='--', label=r"$\epsilon''=$DiHiTI$[\epsilon'_{tap}]$")
    ax2.axhline(0, color='k', lw=0.8)
    ax2.axhspan(min(ht[f'ell{ell}_ht_direct'].min(), ht[f'ell{ell}_ht_dihiti'].min())*1.1, 0,
                color='red', alpha=0.10)
    ax.set_xlabel(r'$(y-y_0)/\lambda$'); ax.set_ylabel(r"$\epsilon'$", color='#1f77b4')
    ax2.set_ylabel(r"$\epsilon''$", color='#8B0000')
    ax.tick_params(axis='y', colors='#1f77b4'); ax2.tick_params(axis='y', colors='#8B0000')
    ax.set_xlim(-4,4); ax.set_title(rf'$\ell={ell}\lambda$', fontsize=10)
    if ell==0.5:
        h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, fontsize=7, loc='upper left')
    ax2.text(0.98,0.03, 'red band = GAIN', transform=ax2.transAxes, ha='right',
             fontsize=7, color='red')
fig.suptitle(r"Exponentially-tapered Lorentzian: $\epsilon'$ and its Hilbert partners "
             r"(both HT constructions go negative)", fontsize=11)
fig.tight_layout(rect=[0,0,1,0.93])
p1=os.path.join(H,'taper_profile_and_HT.png'); fig.savefig(p1,dpi=200)

# ---- Figure 2: reflectance ----
fig,ax=plt.subplots(figsize=(7.2,5.2))
cols=plt.cm.viridis(np.linspace(0.12,0.82,4))
for c,e in zip(cols,['0.5','1','2','4']):
    d=tp[f'alone_ell{e}']; ax.semilogy(d[0],d[1],'-',color=c,lw=1.9,label=rf'tapered, $\ell={e}\lambda$')
d=tp['alone_notaper']; ax.semilogy(d[0],d[1],'-',color='#8B0000',lw=1.9,label='untapered Lorentzian')
ax.semilogy(ls_['PML_pf1'][0],ls_['PML_pf1'][1],'k-',lw=2.2,label='PML pf=1')
ax.semilogy(ls_['PML_pf4'][0],ls_['PML_pf4'][1],'k--',lw=2.0,label='PML pf=4')
ax.set_xlabel(r'$\theta$ (deg)'); ax.set_ylabel(r'Reflectance $|S_{11}|^2$')
ax.set_xlim(0,89); ax.set_ylim(1e-12,1.0); ax.grid(alpha=0.3); ax.legend(fontsize=8,loc='lower right')
ax.set_title(r'Tapered Lorentzian terminator ($L=8\lambda$, $A$=4, $\gamma=\lambda/5$)',fontsize=10)
fig.tight_layout(); p2=os.path.join(H,'R_vs_angle_tapered.png'); fig.savefig(p2,dpi=200)
print('saved:\n ',p1,'\n ',p2)

print('\nR(0) vs taper length:')
for e in ['0.5','1','2','4']: print(f'  ell={e}lam : {tp[f"alone_ell{e}"][1][0]:.3e}')
print(f'  no taper  : {tp["alone_notaper"][1][0]:.3e}')
print(f'  PML pf=1  : {ls_["PML_pf1"][1][0]:.3e}')
print('\nwhere tapered (ell=1lam) beats PML pf=1:')
d=tp['alone_ell1']; P=ls_['PML_pf1']; w=d[0][d[1]<P[1]]
print(f'  {w.min():.0f}-{w.max():.0f} deg' if len(w) else '  nowhere')
