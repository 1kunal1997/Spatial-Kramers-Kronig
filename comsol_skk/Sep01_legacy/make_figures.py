"""Standalone figures -> comsol_skk/figures/   (revision 2)"""
import os, sys, textwrap, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import hilbert
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,ROOT); import tmm_helper as tmm_h
FIG=os.path.join(HERE,'figures'); os.makedirs(FIG,exist_ok=True)
D=np.load(os.path.join(HERE,'master_data.npz'))
plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':0.28,
                     'axes.labelsize':12,'legend.framealpha':0.92,'figure.dpi':110})
NB=1.7
K3=['logistic','lorentzian','tapered']
LBL={'logistic':'logistic (DiHiTI)','lorentzian':'Lorentzian','tapered':'tapered Lorentzian',
     'gain':'tapered + exact HT (gain)'}
C={'logistic':'#1f77b4','lorentzian':'#8B0000','tapered':'#ff7f0e','gain':'#2ca02c'}
STY={'logistic':'-','lorentzian':'-','tapered':'--','gain':'-'}
PC={1:'k',2:'0.35',4:'0.6'}
NEL=sorted(set(int(round(v)) for v in np.geomspace(5,1000,22)))
LGRID=sorted(set(round(float(v),3) for v in np.geomspace(1,100,21)))

def save(fig,name,note):
    fig.text(0.005,-0.015,'\n'.join(textwrap.wrap(note,105)),fontsize=6.8,color='0.45',va='top')
    fig.savefig(os.path.join(FIG,name),dpi=200,bbox_inches='tight'); plt.close(fig); print('  ',name)
def newax(xl=r'$\theta$ (deg)',yl=r'Reflectance  $|S_{11}|^2$',fs=(7.2,5.2)):
    fig,ax=plt.subplots(figsize=fs); ax.set_xlabel(xl); ax.set_ylabel(yl); return fig,ax
def at(d,t):  return d[1][int(np.argmin(np.abs(d[0]-t)))]

# ---------------------------------------------------------------- figure registry
FIGS = {}

def _f01():
    """01 schematic"""
    fig,ax=plt.subplots(figsize=(6.4,5.6))
    ax.add_patch(Rectangle((0,7),3,3,fc='#dce9f7',ec='k',lw=1.4))
    ax.add_patch(Rectangle((0,2),3,5,fc='#ffe0b3',ec='k',lw=1.4))
    ax.add_patch(Rectangle((0,0),3,2,fc='#d8f0d8',ec='k',lw=1.4))
    ax.plot([0,3],[10,10],'b-',lw=4); ax.plot([0,3],[0,0],'k-',lw=6)
    ax.annotate('',xy=(1.5,8.4),xytext=(2.35,9.85),arrowprops=dict(arrowstyle='-|>',lw=2.2,color='crimson'))
    ax.text(1.5,10.35,'periodic Port  (launch + absorb, exact for order 0)',ha='center',fontsize=9,color='b')
    ax.text(1.5,-0.55,'PEC  (worst-case backing: forces the round trip)',ha='center',fontsize=9)
    ax.text(1.5,8.5,r'substrate  $n_b=1.7$',ha='center',fontsize=11)
    ax.text(1.5,4.4,'TERMINATOR\n'+r'$\epsilon(y)$',ha='center',fontsize=11,weight='bold')
    ax.text(1.5,0.9,'lower region (PML or plain)',ha='center',fontsize=9.5)
    ax.text(3.15,5,'Floquet periodic\n(both sides)',fontsize=8.5,va='center',color='0.35')
    ax.text(2.5,9.2,r'$\theta$',fontsize=12,color='crimson')
    ax.set_xlim(-0.9,4.5); ax.set_ylim(-1.1,11); ax.axis('off'); ax.grid(False)
    ax.set_title('Terminator test harness',fontsize=13,weight='bold')
    save(fig,'01_geometry_schematic.png','2D, s-pol (Ez), lam=800nm, period Lx=lam/4 so only diffraction order 0 propagates.')

FIGS['01'] = _f01

def _f02():
    """02 PML baseline"""
    fig,ax=newax()
    for pf in [1,2,4]:
        d=D[f'PML_pf{pf}']; ax.semilogy(d[0],d[1],'-',color=PC[pf],lw=2.2,label=f'PMLfactor={pf}')
    ax.set_ylim(1e-13,1.5); ax.set_xlim(0,89); ax.legend(fontsize=10,loc='lower right')
    ax.set_title('PML baseline',fontsize=13,weight='bold')
    save(fig,'02_pml_baseline.png','PML thickness 2 lam. 800 elements across the PML -- deliberately over-resolved '
         '(fig 3 shows pf=1 converges by ~50 and pf=4 by ~100-200) so nothing in this baseline can be blamed on mesh.')
    if os.path.exists(os.path.join(FIG,'02_pml_angle_cosine_law.png')): os.remove(os.path.join(FIG,'02_pml_angle_cosine_law.png'))

FIGS['02'] = _f02

def _f03():
    """03 / 03b PML elements"""
    for tag,thv,name in [('a',0,'03_pml_R_vs_elements_theta0.png'),('b',80,'03b_pml_R_vs_elements_theta80.png')]:
        fig,ax=newax(xl='elements across the PML',yl='R')
        for pf in [1,2,4]:
            ax.loglog(NEL,[at(D[f'P2_PMLN_pf{pf}_N{n}'],thv) for n in NEL],'o-',color=PC[pf],lw=2,ms=5,
                      label=f'PMLfactor={pf}')
        ax.set_title(rf'$\theta={thv}^{{\circ}}$')
        ax.legend(fontsize=9.5,loc='upper right' if thv==80 else 'lower left')
        save(fig,name,f'A-E "D" (PML). R at theta={thv} deg. PML LENGTH HELD CONSTANT at 2 lam; only the element '
             'count changes. All PMLfactors reach the same floor -- higher pf simply needs more elements to get there.')
    if os.path.exists(os.path.join(FIG,'03_pml_R_vs_elements.png')): os.remove(os.path.join(FIG,'03_pml_R_vs_elements.png'))

FIGS['03'] = _f03

def _f04():
    """04 PML thickness"""
    fig,ax=newax(xl=r'PML thickness / $\lambda$',yl=r'$R$')
    for t in [0,40,80]:
        ax.loglog([1,2,4,8,16],[at(D[f'PMLL_L{L}'],t) for L in [1,2,4,8,16]],'o-',lw=2,ms=6,label=rf'$\theta={t}^\circ$')
    ax.set_ylim(1e-13,1); ax.legend(fontsize=9.5)
    save(fig,'04_pml_R_vs_thickness.png','PMLfactor = 1. MESH HELD CONSTANT at 400 elements for every point; only the '
         'thickness changes, so element density falls 16x across the sweep and R still does not move. COMSOL stretches '
         'x -> x + lambda_typ*f(d/d_max): the profile is NORMALISED by thickness, so only PMLfactor sets absorption.')

FIGS['04'] = _f04

def _f05():
    """05 / 06 profiles (flipped: entry at LEFT)"""
    Lp=8.0
    for part,idx,ylab,name,ttl in [('re',1,r"$\epsilon\,'(y)$",'05_profiles_epsilon_real.png','Real part of the terminator profiles'),
                                   ('im',2,r"$\epsilon\,''(y)$",'06_profiles_epsilon_imag.png','Imaginary part of the terminator profiles')]:
        fig,ax=newax(xl=r'$y/\lambda$',yl=ylab)
        ax.set_yscale('linear')
        for k in K3:
            p=D[f'PROF_{k}']; ax.plot(Lp-p[0],p[idx],STY[k],color=C[k],lw=2.2,label=LBL[k])
        if part=='re':
            ax.axhline(NB**2,color='0.5',ls=':',lw=1.2); ax.text(7.85,NB**2+0.10,r'$\epsilon_b=n_b^2$',fontsize=9,color='0.4',ha='right')
            ax.axhline(1,color='0.5',ls=':',lw=1.2); ax.text(7.85,1.10,'air',fontsize=9,color='0.4',ha='right')
        else:
            ax.axhline(0,color='k',lw=1.0)
        ax.legend(fontsize=9.5); ax.set_xlim(0,8)
        ax.set_title(ttl,fontsize=12.5,weight='bold')
        save(fig,name,'Wave enters from the LEFT (y=0 is the entry face; y increases into the terminator). '
             'Logistic: net step eps_b -> air, so TIR above 36 deg. Lorentzian family: eps_b on BOTH sides -- '
             'no step, no TIR, honest 0-90 deg.')

FIGS['05'] = _f05

def _f07():
    """07 / 13 / 14 matched-mesh comparisons"""
    for Lm,name in [(8,'07_comparison_L8.png'),(2,'13_comparison_L2.png'),(12,'14_comparison_L12.png')]:
        fig,ax=newax(fs=(7.4,5.2))
        for k in K3:
            d=D[f'P2_M{Lm}_{k}']; ax.semilogy(d[0],d[1],STY[k],color=C[k],lw=2.2,label=LBL[k])
        for pf in [1,4]:
            d=D[f'P2_M{Lm}_PML{pf}']
            ax.semilogy(d[0],d[1],'--' if pf==4 else '-',color=PC[pf],lw=2.0,label=f'PML pf={pf}')
        ax.set_ylim(1e-13,1.5); ax.set_xlim(0,89); ax.legend(fontsize=9,loc='lower right')
        ax.set_title(rf'Thickness held at $L={Lm}\lambda$',fontsize=13,weight='bold')
        save(fig,name,'MESH IDENTICAL ON EVERY CURVE: 800 elements in the terminator domain + 800 in the lower '
             'domain, same geometry, same port. The only difference between curves is what fills the terminator. '
             'Note: above 36 deg the logistic is past TIR, where low R is evanescent absorption, not the sKK mechanism.')
    for old in ['07_headline_comparison_L8.png','13_fixed_thickness_L2.png','14_fixed_thickness_L12.png']:
        p=os.path.join(FIG,old)
        if os.path.exists(p): os.remove(p)

FIGS['07'] = _f07

def _f09():
    """09 truncation effects"""
    fig,ax=newax(xl=r'terminator thickness $L/\lambda$',yl='R')
    for k in K3:
        xs=[L for L in LGRID if f'P4_TH_{k}_L{L}' in D.files]
        ax.loglog(xs,[D[f'P4_TH_{k}_L{L}'][1][0] for L in xs],STY[k],color=C[k],lw=2.2,marker='o',ms=4,label=LBL[k])
    ax.axhline(D['P10_PML_pf1'][1][80],color='k',lw=2.2,label='PML pf=1')
    ax.axhline(D['P10_PML_pf4'][1][80],color='0.55',lw=2.2,ls='--',label='PML pf=4')
    ax.legend(fontsize=9,loc='upper right')
    ax.set_title(r'Truncation effects $(\theta=80^{\circ})$',fontsize=13,weight='bold')
    save(fig,'09_truncation_effects.png','A-E "B", evaluated at theta = 80 deg. Lorentzian ~ L^-2 (1/y tails), '
         'logistic ~ L^-4 (exponential tails). '
         'ALL profiles use the SAME fixed feature width lam/5. Element count is NOT constant along x: '
         'N = 400 for L <= 8 lam, then 50 per lam (5000 at L = 100 lam), i.e. element size fixed at '
         '0.02 lam = 10 elements across the feature. Same rule for all three profiles, so any vertical '
         'comparison is at identical mesh. Converged: doubling AND quadrupling N changes R by 0.00% at '
         'L = 1, 2, 8, 32 and 100 lam. Below L ~ 2 lam the logistic transition does not fully fit in the '
         'domain (14% short of air at L = 1 lam), which understates it slightly there. Both PML lines are '
         'thickness-independent (fig 4). PML lines are the pinned measurement (P10, same harness as every '
         'other PML number here): pf=1 = 1.300e-2, pf=4 = 9.927e-7 at theta=80. The logistic CATCHES the '
         'tuned pf=4 PML at L = 100 lam (8.34e-7 vs 9.93e-7) -- call it a tie, since the PML floor in this '
         'harness carries a factor ~2 systematic.')
    if os.path.exists(os.path.join(FIG,'09_R_vs_thickness.png')): os.remove(os.path.join(FIG,'09_R_vs_thickness.png'))

FIGS['09'] = _f09

def _f10():
    """10 equal-cost, one figure per angle"""
    for ti,(thv,tag) in enumerate([(0,'00'),(45,'45'),(80,'80'),(89,'89')]):
        fig,ax=newax(xl='elements across the absorbing region',yl='R',fs=(8.4,5.2))
        for k in K3:
            ax.loglog(NEL,[D[f'P7_EL_{k}_N{n}'][1][ti] for n in NEL],STY[k],color=C[k],lw=2.2,marker='o',ms=4,label=LBL[k])
        for pf in [1,4]:
            ax.loglog(NEL,[D[f'P7_PML_pf{pf}_N{n}'][1][ti] for n in NEL],':',color=PC[pf],lw=2.2,marker='s',ms=4,
                      label=f'PML pf={pf}')
        ax.axvline(5,color='crimson',ls=':',lw=1.4)
        ax.legend(fontsize=8.5,loc='center left',bbox_to_anchor=(1.02,0.5))
        ax.set_title(rf'Equal-cost comparison ($\theta={thv}^{{\circ}}$)',fontsize=13,weight='bold')
        save(fig,f'10_equal_cost_theta{tag}.png','A-E "D". THE mesh-held-constant figure, at theta = '
             f'{thv} deg. sKK profiles at L=8 lam (feature width lam/5), PML at L=2 lam -- extra thickness '
             'does nothing for a PML (fig 4). All five curves share one geometry, port and solver; only the '
             'element count across the absorbing region varies. Read vertically: at any element budget, '
             'which terminator gives the lowest R. Red line = 5 elements, a common default.')

FIGS['10'] = _f10

def _f15():
    """15 amplitude"""
    fig,ax=newax(xl='amplitude parameter',yl=r'$R(\theta=0)$')
    AS=[1,2,4,8,16]
    ax.loglog(AS,[D[f'PAR_lorA{a}'][1][0] for a in AS],'o-',color=C['lorentzian'],lw=2,ms=6,label=r'Lorentzian, $A$')
    ax.loglog(AS,[D[f'PAR_tapA{a}'][1][0] for a in AS],'o--',color=C['tapered'],lw=2,ms=6,label=r'tapered Lorentzian, $A$')
    xs=np.array([1,16.]); ax.loglog(xs,2e-5*xs**2,':',color='0.4',lw=1.6,label=r'$\propto A^{2}$')
    ax.axhline(D['PML_pf1'][1][0],color='k',lw=2,label='PML pf=1')
    ax.set_ylim(1e-13,1e-1); ax.legend(fontsize=9,loc='lower right')
    ax.set_title(r'Length held constant ($L=8\lambda$): sweep amplitude',fontsize=12.5,weight='bold')
    save(fig,'15_param_sweep_amplitude.png','Advisor question: "did you hold length constant and sweep amplitude?" '
         'R ~ A^2 exactly, as the truncation step Delta eps\' = 2*A*gam/L predicts.')

FIGS['15'] = _f15

def _f16():
    """16 width"""
    fig,ax=newax(xl=r'width parameter / $\lambda$',yl=r'$R(\theta=0)$')
    gs=[10,5,2.5,1.25]; ws=[80,40,20,10]; es=[0.25,0.5,1,2,4]
    ax.loglog([1/g for g in gs],[D[f'PAR_lorG{g}'][1][0] for g in gs],'o-',color=C['lorentzian'],lw=2,ms=6,label=r'Lorentzian $\gamma$')
    ax.loglog([8/w for w in ws],[D[f'PAR_logW{w}'][1][0] for w in ws],'o-',color=C['logistic'],lw=2,ms=6,label=r'logistic $w$')
    ax.loglog(es,[D[f'PAR_tapE{e}'][1][0] for e in es],'o--',color=C['tapered'],lw=2,ms=6,label=r'taper length $\ell$')
    ax.axhline(D['PML_pf1'][1][0],color='k',lw=2,label='PML pf=1')
    ax.set_ylim(1e-13,1e-1); ax.legend(fontsize=9,loc='lower right')
    ax.set_title(r'Length held constant ($L=8\lambda$): sweep width',fontsize=12.5,weight='bold')
    save(fig,'16_param_sweep_width.png','Advisor question: "did you hold length constant and sweep width?" '
         'The taper has an INTERIOR optimum near ell ~ 0.5-1 lam: harder taper shrinks the truncation step but grows '
         'the KK mismatch, because eps\'\' is no longer the Hilbert partner of the eps\' you actually have.')

FIGS['16'] = _f16

def _f17():
    """17 theorems"""
    lam=0.8; eb=NB**2; A=4.0; gam=lam/5
    W=400*lam; N=2**20; u=np.linspace(-W/2,W/2,N); Dd=u**2+gam**2
    dre=-A*gam*u/Dd; im=A*gam**2/Dd
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12.4,5.4))
    a1.plot(u/lam,im,color='#8B0000',lw=2,label=r"$\epsilon\,''$ (Lorentzian)")
    a1.plot(u/lam,dre,color='#1f77b4',lw=1.7,label=r"$\epsilon\,'-\epsilon_b$")
    a1.set_xlim(-6,6); a1.set_xlabel(r'$u/\lambda$'); a1.legend(fontsize=9.5)
    a1.set_title('sum rule for the untruncated Lorentzian',fontsize=11.5,weight='bold')
    a1.text(0.03,0.03,"$1/u$ tail coefficient of $\\epsilon\\,'$ = %.4f\n$(1/\\pi)\\int\\epsilon\\,''\\,dy$ = %.4f"
            %(A*gam,np.trapezoid(im,u)/np.pi), transform=a1.transAxes,fontsize=10.5,va='bottom',
            bbox=dict(fc='lightyellow',ec='0.6',alpha=0.95))
    for ell,cc in zip([0.5,1.0,2.0],['#2ca02c','#ff7f0e','#9467bd']):
        ht=np.imag(hilbert(dre/np.cosh(u/(ell*lam))))
        a2.plot(u/lam,ht,color=cc,lw=1.7,label=rf'HT[$\epsilon\,\prime$ tapered], $\ell={ell}\lambda$,  $\int$={np.trapezoid(ht,u):.1e}')
    a2.plot(u/lam,im,color='#8B0000',lw=2.2,ls='--',label=r"Lorentzian $\epsilon\,''$,  $\int$=%.3f"%np.trapezoid(im,u))
    a2.axhline(0,color='k',lw=1); a2.axhspan(-0.45,0,color='crimson',alpha=0.12)
    a2.text(5.6,-0.33,'GAIN',ha='right',color='crimson',fontsize=10,weight='bold')
    a2.set_xlim(-6,6); a2.set_xlabel(r'$u/\lambda$'); a2.legend(fontsize=7.6,loc='upper left')
    a2.set_title(r'exponential tails $\Rightarrow$ $\epsilon\,\prime$ becomes $L^1$ $\Rightarrow$ $\int$HT$=0$',fontsize=11.5)
    fig.tight_layout()
    save(fig,'17_sum_rule_and_gain_theorem.png','WHY the design space has the shape it does. LEFT: the 1/y tail '
         'coefficient EQUALS (1/pi)*Int eps\'\' dy -- absorption and slow tails are the same number. RIGHT: give eps\' '
         'exponential tails and it becomes absolutely integrable, so Int H[f] = 0 identically, forcing eps\'\' negative. '
         'Plus Paley-Wiener: compact support and a one-sided spectrum are incompatible, so truncation ALWAYS leaks.')

FIGS['17'] = _f17

def _f18():
    """18 gain (corrected)"""
    fig,(b1,b2)=plt.subplots(1,2,figsize=(12.4,5.0))
    p=D['P2_PROF_gain']; b1b=b1.twinx()
    b1.plot(Lp-p[0],p[1],color='#1f77b4',lw=2.2,label=r"$\epsilon\,'$")
    b1b.plot(Lp-p[0],p[2],color='#8B0000',lw=2.2,label=r"$\epsilon\,''$")
    b1b.axhline(0,color='k',lw=1); b1b.axhspan(p[2].min()*1.15,0,color='crimson',alpha=0.13)
    b1b.text(7.8,p[2].min()*0.5,'GAIN',ha='right',color='crimson',fontsize=10,weight='bold')
    b1.set_xlabel(r'$y/\lambda$'); b1.set_ylabel(r"$\epsilon\,'$",color='#1f77b4')
    b1b.set_ylabel(r"$\epsilon\,''$",color='#8B0000'); b1b.grid(False)
    b1.tick_params(axis='y',colors='#1f77b4'); b1b.tick_params(axis='y',colors='#8B0000')
    b1.set_xlim(0,8); b1.set_title(r'tapered $\epsilon\,\prime$ with its exact HT partner',fontsize=11.5,weight='bold')
    h1,l1=b1.get_legend_handles_labels(); h2,l2=b1b.get_legend_handles_labels()
    b1.legend(h1+h2,l1+l2,fontsize=9.5,loc='upper right')
    d=D['P2_gain_fine']; b2.semilogy(d[0],d[1],'-',color=C['gain'],lw=2.4,label='tapered + exact HT (gain)')
    d=D['FINE_tapered']; b2.semilogy(d[0],d[1],'--',color=C['tapered'],lw=2.0,label=r"tapered, passive (same $\epsilon\,'$)")
    d=D['PML_pf1']; b2.semilogy(d[0],d[1],'k-',lw=2.0,label='PML pf=1')
    b2.axhline(1,color='crimson',ls='--',lw=1.6)
    b2.text(1.5,1.5,'$R=1$: amplification',color='crimson',fontsize=9)
    b2.set_ylim(1e-13,1e2); b2.set_xlim(0,89); b2.set_xlabel(r'$\theta$ (deg)')
    b2.set_ylabel(r'Reflectance $|S_{11}|^2$'); b2.legend(fontsize=8.5,loc='lower right')
    b2.set_title('and what it does',fontsize=11.5,weight='bold')
    fig.tight_layout()
    save(fig,'18_gain_terminator.png','A PML is already unphysical, so gain is fair game -- and COMSOL solves it '
         'cleanly (no divergence). It still fails: the sum rule forces Int eps\'\' = 0, so the exact-HT profile is '
         'net-NEUTRAL and cannot absorb. It is WORSE than the passive tapered profile with the identical eps\'.')
    print('\nall figures written to',FIG)

FIGS['18'] = _f18


def _f19():
    """19 scale sweep (c): L and feature width scaled together"""
    LG=[round(float(v),3) for v in np.geomspace(0.5,100,18)]
    for ti,(thv,tag) in enumerate([(0,'00'),(45,'45'),(80,'80'),(89,'89')]):
        fig,ax=newax(xl=r'terminator size  $L/\lambda$   (feature width $=L/80$)',
                     yl='R',fs=(8.4,5.2))
        for k in K3:
            xs=[L for L in LG if f'P8_r80_{k}_L{L}' in D.files]
            ax.loglog(xs,[D[f'P8_r80_{k}_L{L}'][1][ti] for L in xs],STY[k],color=C[k],
                      lw=2.2,marker='o',ms=4,label=LBL[k])
        for pf in [1,4]:
            xs=[L for L in LG if f'P8_PML_pf{pf}_L{L}' in D.files]
            ax.loglog(xs,[D[f'P8_PML_pf{pf}_L{L}'][1][ti] for L in xs],':',color=PC[pf],
                      lw=2.2,marker='s',ms=4,label=f'PML pf={pf}')
        ax.legend(fontsize=8.5,loc='center left',bbox_to_anchor=(1.02,0.5))
        ax.set_title(rf'Size relative to wavelength ($\theta={thv}^{{\circ}}$)',fontsize=13,weight='bold')
        save(fig,f'19_scale_sweep_theta{tag}.png','Sweep (c): L and the feature width scale TOGETHER at '
             'fixed L/w = 80, so TRUNCATION IS CONSTANT by construction and this isolates absolute size vs '
             'wavelength. Because the media are non-dispersive, this is equivalent to a WAVELENGTH SWEEP at '
             'fixed geometry (the d/lambda axis). Elements scale with L, so resolution per wavelength and per '
             'feature are both held constant. The PML thickness is scaled the same way.')

FIGS['19'] = _f19

def _f20():
    """20 ratio collapse test"""
    LG=[round(float(v),3) for v in np.geomspace(0.5,100,18)]
    fig,axs=plt.subplots(1,2,figsize=(12.6,5.0))
    for ax,(thv,ti) in zip(axs,[(0,0),(80,2)]):
        for k in K3:
            for ratio,ls,al in [(40,':',0.55),(80,'-',1.0)]:
                xs=[L for L in LG if f'P8_r{ratio}_{k}_L{L}' in D.files]
                ax.loglog(xs,[D[f'P8_r{ratio}_{k}_L{L}'][1][ti] for L in xs],ls,color=C[k],
                          lw=2.0,alpha=al,marker='o' if ratio==80 else None,ms=3,
                          label=f'{LBL[k]}, $L/w$={ratio}')
        ax.set_xlabel(r'$L/\lambda$'); ax.set_ylabel(rf'$R(\theta={thv}^\circ)$')
        ax.set_title(rf'$\theta={thv}^\circ$',fontsize=11.5)
        if thv==0: ax.legend(fontsize=7.2,loc='lower left')
    fig.tight_layout()
    save(fig,'20_ratio_collapse.png','Does the sweep-(c) curve depend on the L/w ratio? If truncation is '
         'already negligible at L/w = 40, the two ratios should collapse onto one another and the curve is a '
         'pure size-vs-wavelength effect. Where they separate, truncation is still contributing.')

FIGS['20'] = _f20


# aliases: some functions emit more than one figure

# ---------------------------------------------------------------- patch13 / patch14
GC = {'uniform':'#1f77b4', 'ctr':'#8B0000', 'ent':'#ff7f0e'}
GL = {'uniform':'uniform (equidistant)', 'ctr':'centre-dense (symmetric+reverse)',
      'ent':'entry-dense (reverse)'}
RATIOS13 = [5, 20, 100]

def _f29():
    """29 mesh grading, one figure per element ratio"""
    for r in RATIOS13:
        fig,ax=newax(fs=(7.4,5.2))
        d=D['P13_uniform_N800']
        ax.semilogy(d[0],d[1],':',color='0.55',lw=1.8,label='uniform, 800 elem. (converged)')
        for tag in ['uniform','ctr','ent']:
            key='P13_uniform_N50' if tag=='uniform' else f'P13_{tag}_r{r}_N50'
            d=D[key]; ax.semilogy(d[0],d[1],'-',color=GC[tag],lw=2.2,label=GL[tag])
        ax.set_ylim(1e-13,1.5); ax.set_xlim(0,89); ax.legend(fontsize=9,loc='lower right')
        ax.set_title(rf'Mesh grading, tapered Lorentzian ($L=10\lambda$, 50 elements, '
                     rf'ratio {r})',fontsize=12.5,weight='bold')
        save(fig,f'29_mesh_grading_r{r}.png','Same model, same geometry, same 50 elements across the '
             f'terminator -- only their DISTRIBUTION changes. Element ratio {r} = size of the largest '
             'element / smallest. COMSOL Distribution: type=predefined, geometric; centre-dense adds '
             'symmetric+reverse, entry-dense adds reverse only (fine at the entry face, coarsening '
             'toward the PEC). Grey dotted = the converged answer (800 uniform elements), i.e. the '
             'target every 50-element curve is trying to hit. Tapered Lorentzian, gam = lam/5, '
             'ell = 0.5 lam, A = 4, n_b = 1.7.')

FIGS['29'] = _f29

def _f30():
    """30 turned-logistic profile, air background"""
    p=D['P14_PROF']; Lp=p[0][-1]
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    ax.plot(Lp-p[0],p[1],'-',color='#1f77b4',lw=2.4)
    ax.set_xlabel(r'$y/\lambda$'); ax.set_ylabel(r"$\epsilon\,'(y)$",color='#1f77b4')
    ax.tick_params(axis='y',labelcolor='#1f77b4')
    ax.axhline(1,color='0.5',ls=':',lw=1.2); ax.text(Lp*0.985,1.02,'air',fontsize=9,color='0.4',ha='right')
    ax2=ax.twinx(); ax2.plot(Lp-p[0],p[2],'-',color='#8B0000',lw=2.4)
    ax2.set_ylabel(r"$\epsilon\,''(y)$",color='#8B0000'); ax2.tick_params(axis='y',labelcolor='#8B0000')
    ax2.grid(False); ax.set_xlim(0,Lp)
    ax.set_title('Turned logistic in an air background',fontsize=12.5,weight='bold')
    save(fig,'30_air_flip_profile.png',"Wave enters from the LEFT. eps'' carries the logistic (lossless "
         "-> lossy, 0 -> 1); eps' is its Hilbert partner, shifted so eps' = 1 (air) at both ends -- the "
         "standard sKK logistic with the two parts exchanged, i.e. the same one-sided analytic function "
         "multiplied by e^{-i pi/2}. Matched AND lossless at the entry face, constant loss at the PEC. "
         "eps' never goes below 1, so there is no critical angle: the whole 0-90 deg range is sKK, not "
         "evanescent absorption. L = 10 lam, feature width lam/5.")

FIGS['30'] = _f30

def _f31():
    """31 turned logistic vs PML, both in air"""
    LS=[2,5,10,32,100]; cols=plt.cm.viridis(np.linspace(0.05,0.85,len(LS)))
    fig,ax=newax(fs=(7.6,5.2))
    for L,c in zip(LS,cols):
        d=D[f'P14_AIR_L{L}']
        ax.semilogy(d[0],d[1],'-',color=c,lw=2.2,label=rf'turned logistic, $L={L}\lambda$')
    for pf in [1,4]:
        d=D[f'P14_PML_pf{pf}']
        ax.semilogy(d[0],d[1],'--' if pf==4 else '-',color=PC[pf],lw=2.0,label=f'PML pf={pf}')
    ax.set_ylim(1e-14,1.5); ax.set_xlim(0,89)
    ax.legend(fontsize=8.5,loc='center left',bbox_to_anchor=(1.02,0.5))
    ax.set_title('Turned logistic vs PML, air background',fontsize=13,weight='bold')
    save(fig,'31_air_flip_R_vs_angle.png','AIR background (n_bg = 1) -- a PML terminating vacuum, its '
         'home turf, and the case a PML user actually has. Every earlier figure in this folder is at '
         'n_b = 1.7 and is NOT comparable to this one. Element size held at 0.02 lam on every sKK curve '
         '(N = 50 per lam, 400 minimum); PML at 2 lam with 800 elements. No TIR anywhere on these curves: '
         "eps' >= 1 everywhere, so the full 0-89 deg range is the sKK mechanism.")

FIGS['31'] = _f31

ALIAS = {'06':'05', '13':'07', '14':'07', '03b':'03'}
NOTE  = {'29':'-> one file per element ratio', '05':'-> 05 and 06', '07':'-> 07, 13 and 14', '03':'-> 03 and 03b'}

def _cleanup():
    for old in ['08_grazing_zoom.png','11_stacking_ratio.png','12_backing_isolation.png',
                '02_pml_angle_cosine_law.png','03_pml_R_vs_elements.png','09_R_vs_thickness.png',
                '10_R_vs_elements.png','07_headline_comparison_L8.png',
                '13_fixed_thickness_L2.png','14_fixed_thickness_L12.png']:
        q=os.path.join(FIG,old)
        if os.path.exists(q): os.remove(q); print('   removed',old)

if __name__ == '__main__':
    args=[a for a in sys.argv[1:] if not a.startswith('-')]
    if '--list' in sys.argv:
        print('available figures:')
        for k in FIGS: print(f'   {k:5s} {NOTE.get(k,"")}')
        print('   aliases:', ', '.join(f'{a}->{b}' for a,b in ALIAS.items())); sys.exit()
    _cleanup()
    todo = [ALIAS.get(a,a) for a in args] if args else list(FIGS)
    todo = list(dict.fromkeys(todo))
    bad=[a for a in todo if a not in FIGS]
    if bad: print('unknown:',bad,'\n  try --list'); sys.exit(1)
    print('writing figures:')
    for k in todo: FIGS[k]()
    print('\ndone ->',FIG)
