import sys; sys.path.insert(0,"/Users/kunal/Documents/Spatial KK Project")
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import lsq_linear
import tmm_helper as tmm_h
BLUE,RED,ORANGE = '#1f77b4','#8B0000','#ff7f0e'
plt.rcParams.update({'font.size':11,'mathtext.fontset':'cm','axes.linewidth':1.1,
                     'lines.linewidth':2.0,'savefig.dpi':160,'savefig.bbox':'tight'})
OUT="/Users/kunal/Documents/Spatial KK Project/theory/figures/2026Aug12_passive_constructor"
nb,L,N=1.7,2.5,2048; xx=np.linspace(-L,L,N); A_=nb**2-1.0
def H(f): return np.imag(hilbert(f))
def dihiti(e):
    n=len(e); c=cumulative_trapezoid(H(np.gradient(e,xx)),xx,initial=0)
    return c-np.linspace(c[0],c[-1],n)
K=60; h=2.2*(2*L)/(K+1); ctr=np.linspace(-L+h,L-h,K)
Phi=np.zeros((N,K))
for k,xc in enumerate(ctr):
    m=np.abs(xx-xc)<h; Phi[m,k]=0.5*(1+np.cos(np.pi*(xx[m]-xc)/h))
M=np.column_stack([-H(Phi[:,k]) for k in range(K)])
AA=np.column_stack([M,np.ones(N),xx])
lo=np.array([0.]*K+[-np.inf]*2); hi=np.full(K+2,np.inf)
def construct(t):
    s=lsq_linear(AA,t,bounds=(lo,hi),max_iter=800); return Phi@s.x[:K], AA@s.x
def g(x0,s): return np.exp(-((xx-x0)/s)**2)
def stack(er,ei,nl=400):
    i=np.linspace(0,N-1,nl).astype(int)
    return ([nb]+list(np.sqrt(er[i]+1j*ei[i]))+[1.0],[np.inf]+list(np.full(nl,2*L/nl))+[np.inf])
logi=tmm_h.logistic(xx,4.0,nb)

# ---------------- FIG 1: constructor reproduces DiHiTI on the logistic
ei_c,er_c=construct(logi); ei_d=dihiti(logi)
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
ax[0].plot(xx,logi,color=BLUE,label="target $\\epsilon'$")
ax[0].plot(xx,er_c,'--',color='k',lw=1.4,label="constructed $\\epsilon'$")
ax[0].set_xlabel('x (μm)'); ax[0].set_ylabel("$\\epsilon'(x)$",color=BLUE); ax[0].legend(fontsize=9)
ax[1].plot(xx,ei_d,color=RED,label="DiHiTI $\\epsilon''$")
ax[1].plot(xx,ei_c,'--',color='k',lw=1.4,label="constructed $\\epsilon''$")
ax[1].axhline(0,color='gray',ls=':',lw=.8)
ax[1].set_xlabel('x (μm)'); ax[1].set_ylabel("$\\epsilon''(x)$",color=RED); ax[1].legend(fontsize=9)
for a,l in zip(ax,'ab'): a.text(-0.14,1.0,f'$\\mathbf{{{l}}}$',transform=a.transAxes,fontsize=14,va='top',ha='right')
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_constructor_validation.png'); plt.close(fig)

# ---------------- FIG 2: repair gallery
cases=[('logistic+bump  $\\alpha$=0.5',logi+0.5*A_*g(0,.6)),
       ('logistic+bump  $\\alpha$=1.0',logi+1.0*A_*g(0,.6)),
       ('opposite bumps $\\alpha$=1.0',logi+1.0*(A_*g(-.9,.45)-A_*g(.9,.45))),
       ('triangle  s=2.0',1.0+2.0*(np.sqrt(L**2+.15**2)-np.sqrt(xx**2+.15**2)))]
fig,axes=plt.subplots(2,4,figsize=(16,6.4))
for j,(lb,t) in enumerate(cases):
    ed=dihiti(t); ec,rc=construct(t)
    a0=axes[0,j]; a0.plot(xx,t,color=BLUE,label='target'); a0.plot(xx,rc,'--',color='k',lw=1.3,label='constructed')
    a0.set_title(lb,fontsize=10); a0.set_xlabel('x (μm)')
    if j==0: a0.set_ylabel("$\\epsilon'(x)$",color=BLUE)
    a0.legend(fontsize=8)
    a1=axes[1,j]; a1.plot(xx,ed,color=RED,label='DiHiTI'); a1.plot(xx,ec,'--',color='k',lw=1.3,label='constructed')
    a1.fill_between(xx,ed,0,where=ed<0,color='red',alpha=.35)
    a1.axhline(0,color='gray',ls=':',lw=.8); a1.set_xlabel('x (μm)')
    a1.set_title(f"min $\\epsilon''$: DiHiTI {ed.min():.3f} → constr {ec.min():.3f}",fontsize=9)
    if j==0: a1.set_ylabel("$\\epsilon''(x)$",color=RED)
    a1.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_repair_gallery.png'); plt.close(fig)

# ---------------- FIG 3: price of passivity
alphas=np.linspace(0,2.0,11); lam=2*L
gains,Rd_,Rc_=[],[],[]
for al in alphas:
    t=logi+al*A_*g(0,.6); ed=dihiti(t); ec,rc=construct(t)
    nd,dd=stack(t,ed); nc,dc=stack(rc,ec)
    _,Rd,_=tmm_h.TRA(nd,dd,lamb=lam,angle=0,pol='s')
    _,Rc,_=tmm_h.TRA(nc,dc,lamb=lam,angle=0,pol='s')
    gains.append(-ed.min()); Rd_.append(Rd); Rc_.append(Rc)
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
ax[0].plot(alphas,gains,'o-',color=RED); ax[0].set_xlabel(r'bump amplitude $\alpha$')
ax[0].set_ylabel("gain in DiHiTI profile  $-\\min\\epsilon''$",color=RED)
ax[1].semilogy(alphas,Rd_,'o-',color=ORANGE,label='DiHiTI (has gain)')
ax[1].semilogy(alphas,Rc_,'s--',color='k',label='constructed (passive)')
ax[1].axhline(6.722e-2,color='gray',ls=':',label='bare interface')
ax[1].set_xlabel(r'bump amplitude $\alpha$'); ax[1].set_ylabel('Reflectance  $R$'); ax[1].legend(fontsize=9)
for a,l in zip(ax,'ab'): a.text(-0.16,1.0,f'$\\mathbf{{{l}}}$',transform=a.transAxes,fontsize=14,va='top',ha='right')
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_price_of_passivity.png'); plt.close(fig)
print("saved 3 figures to",OUT)
