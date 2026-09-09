import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=os.path.dirname(os.path.abspath(__file__))
d=np.load(os.path.join(H,'lorentzian_stacked.npz'),allow_pickle=True)

fig,(a0,a1)=plt.subplots(1,2,figsize=(12.5,5.0))
# left: truncation-length sweep, alone vs stacked
cols=plt.cm.viridis(np.linspace(0.1,0.85,4))
for c,L in zip(cols,[2,4,8,16]):
    a0.semilogy(d[f'alone_L{L}'][0],d[f'alone_L{L}'][1],'-',color=c,lw=1.8,label=rf'alone, $L={L}\lambda$')
    a0.semilogy(d[f'stack_L{L}'][0],d[f'stack_L{L}'][1],'--',color=c,lw=1.0)
a0.semilogy(d['PML_pf1'][0],d['PML_pf1'][1],'k-',lw=2.2,label='PML pf=1')
a0.semilogy(d['PML_pf4'][0],d['PML_pf4'][1],'k--',lw=2.0,label='PML pf=4')
a0.set_title(r'Lorentzian sKK: truncation length ($A$=4, $\gamma=\lambda/5$)'+'\nsolid = alone (PEC), dashed = stacked (+PML)',fontsize=9.5)
# right: A sweep + constant A*gam
for A,c in zip([1,2,4,8],plt.cm.plasma(np.linspace(0.1,0.8,4))):
    k=f'stack_A{A}' if A!=4 else 'stack_L8'
    a1.semilogy(d[k][0],d[k][1],'-',color=c,lw=1.8,label=rf'$A$={A}, $\gamma=\lambda/5$')
for A,ls in [(8,(0,(1,1))),(1,(0,(4,2)))]:
    a1.semilogy(d[f'const_Ag_A{A}'][0],d[f'const_Ag_A{A}'][1],ls=ls,color='0.45',lw=1.4,
                label=rf'$A$={A}, $A\gamma$ fixed')
a1.semilogy(d['PML_pf1'][0],d['PML_pf1'][1],'k-',lw=2.2,label='PML pf=1')
a1.semilogy(d['PML_pf4'][0],d['PML_pf4'][1],'k--',lw=2.0,label='PML pf=4')
a1.set_title(r'Lorentzian sKK: amplitude ($L=8\lambda$, stacked)',fontsize=9.5)
for a in (a0,a1):
    a.set_xlabel(r'$\theta$ (deg)'); a.set_ylabel(r'Reflectance $|S_{11}|^2$')
    a.set_xlim(0,89); a.set_ylim(1e-12,1.0); a.grid(alpha=0.3); a.legend(fontsize=7.5,loc='lower right')
fig.suptitle(r'Lorentzian terminator, $\epsilon_b=n_b^2$ on both sides — no index step, no TIR, honest 0–90°',
             fontsize=10.5)
fig.tight_layout(rect=[0,0,1,0.95])
p=os.path.join(H,'R_vs_angle_lorentzian_stacked.png'); fig.savefig(p,dpi=200); print('saved',p)

print('\ntruncation scaling (alone, A=4, gam=lam/5), R(0) vs L:')
prev=None
for L in [2,4,8,16]:
    r=d[f'alone_L{L}'][1][0]
    print(f'  L={L:>2}lam  R(0)={r:.3e}' + (f'   ratio vs prev = {prev/r:.2f}' if prev else ''))
    prev=r
print('\namplitude scaling (stacked, L=8lam), R(0) vs A:')
for A in [1,2,8]:
    print(f'  A={A}  R(0)={d[f"stack_A{A}"][1][0]:.3e}')
print(f'  A=4  R(0)={d["stack_L8"][1][0]:.3e}')
print('\nstacked vs alone (does the PML behind help the Lorentzian?):')
for L in [2,4,8,16]:
    print(f'  L={L:>2}lam  alone={d[f"alone_L{L}"][1][0]:.3e}  stacked={d[f"stack_L{L}"][1][0]:.3e}')
