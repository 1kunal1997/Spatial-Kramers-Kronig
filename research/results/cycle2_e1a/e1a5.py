"""E1(a) v5 -- CORRECT periodization (drop the duplicated endpoint) + spectral engine.
Only then is the decimation error pure aliasing and B3 provable."""
import numpy as np, timeit, json, os
from scipy.signal import hilbert
from e1a3 import ramp, tail_bound, true_tail, build_cases, imin, dihiti, L, A, FRAC
OUT=os.path.dirname(os.path.abspath(__file__))

def oh(u,x):
    p=ramp(u,x)[:-1]; v=np.imag(hilbert(p)); v=v-v[0]; return np.append(v,v[0])

def main(N=2**16, ns=(256,512,1024,2048,4096), ks=(2,3,4,6)):
    x=np.linspace(-L,L,N); cases,kind=build_cases(x)
    keys=[k for k in cases if abs(cases[k][0]-cases[k][-1])>1e-12]
    u0=cases['logistic w=0.05']
    T=lambda f: min(timeit.repeat(f,number=10,repeat=7))/10
    t_B0=T(lambda: imin(dihiti(u0,x),x)); t_B1=T(lambda: imin(oh(u0,x),x))
    print(f"B0 dihiti {t_B0*1e6:.1f}us   B1 onehilb(correct period) {t_B1*1e6:.1f}us")
    print(f"agreement B0 vs B1 at N=2^16, logistic w=0.05: "
          f"{imin(dihiti(u0,x),x):.8e} vs {imin(oh(u0,x),x):.8e}")
    print(f"\n{'n':>6s} {'k':>2s} {'B3viol':>7s} {'maxRatio':>9s} {'cert%':>7s} {'phys%':>7s} {'adv%':>6s} "
          f"{'false':>6s} {'t_bnd':>7s} {'t_dec':>7s} {'E[t]':>8s} {'vsB0':>6s} {'vsB1':>6s}")
    rows=[]
    tr={k:imin(oh(cases[k],x),x) for k in keys}
    for n in ns:
        M=n//2; s=N//n
        tdec=T(lambda s=s: imin(oh(u0[::s],x[::s]),x[::s]))
        cd={k:imin(oh(cases[k][::s],x[::s]),x[::s]) for k in keys}
        for kk in ks:
            tbnd=T(lambda kk=kk,M=M: tail_bound(u0,x,M,kk))
            viol=nc=ncp=nca=nf=0; npz=nadv=0; worst=0.0
            for key in keys:
                u=cases[key]; D=abs(u[0]-u[-1])
                b=4*tail_bound(u,x,M,kk)
                r=abs(cd[key]-tr[key])/max(b,1e-300); worst=max(worst,r)
                if r>1.001: viol+=1
                cert=abs(cd[key])>b
                if kind[key]=='phys': npz+=1; ncp+=cert
                else: nadv+=1; nca+=cert
                nc+=cert
                if cert and ((tr[key]>=0)!=(cd[key]>=0)): nf+=1
            tot=npz+nadv; pc=nc/tot; Et=tbnd+tdec+(1-pc)*t_B1
            print(f"{n:6d} {kk:2d} {viol:7d} {worst:9.2e} {100*pc:6.1f}% {100*ncp/npz:6.1f}% "
                  f"{100*nca/max(nadv,1):5.1f}% {nf:6d} {tbnd*1e6:6.1f}u {tdec*1e6:6.1f}u "
                  f"{Et*1e6:7.1f}u {t_B0/Et:6.2f} {t_B1/Et:6.2f}")
            rows.append(dict(n=n,k=kk,viol=viol,worst=worst,cert=pc,cert_phys=ncp/npz,
                             cert_adv=nca/max(nadv,1),false=nf,Et=Et,vsB0=t_B0/Et,vsB1=t_B1/Et))
    json.dump(dict(N=N,rows=rows,t_B0=t_B0,t_B1=t_B1,nkeys=len(keys)),
              open(os.path.join(OUT,f"e1a5_N{N}.json"),"w"),indent=1)

if __name__=="__main__": main()
