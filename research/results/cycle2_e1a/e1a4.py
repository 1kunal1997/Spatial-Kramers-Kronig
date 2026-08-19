"""E1(a) v4 -- spectral engine, so the ONLY decimation error is aliasing and B3 is provable.

truth   = onehilb at N   (spectrally exact Hilbert of the ramp-periodized profile)
cand    = onehilb at n   (same transform, decimated)
cert    = |min_cand| > 4 * B2(k, M=n/2)   [B2 verified sound in e1a3.py]
"""
import numpy as np, timeit, json, os
from e1a3 import ramp, tail_bound, true_tail, build_cases, imin, dihiti, L, A, FRAC
from e1a import onehilb
OUT = os.path.dirname(os.path.abspath(__file__))

def main(N=2**16, ns=(256,512,1024,2048,4096), ks=(2,3,4,6)):
    x = np.linspace(-L,L,N); cases, kind = build_cases(x); keys=[k for k in cases
                                                                if abs(cases[k][0]-cases[k][-1])>1e-12]
    print("=== B3 soundness audit (spectral engine): actual err vs 4*B2 ===")
    print(f"{'case':16s} {'M':>5s} {'k':>2s} {'true_tail':>10s} {'B2':>10s} {'err':>10s} {'4B2':>10s} {'ok':>4s}")
    bad=0; audit=[]
    for key in ['logistic w=0.05','logistic w=0.3','bumps a=1.0','rand3','cos m=257','cos m=1024']:
        u=cases[key]; D=abs(u[0]-u[-1]); tr=imin(onehilb(u,x),x)
        for n in (512,2048):
            M=n//2; s=N//n; cd=imin(onehilb(u[::s],x[::s]),x[::s]); err=abs(cd-tr)/D
            tt=true_tail(u,x,M)/D
            for kk in ks:
                b=tail_bound(u,x,M,kk)/D; ok=(b>=tt*0.999) and (4*b>=err*0.999); bad+=not ok
                print(f"{key:16s} {M:5d} {kk:2d} {tt:10.3e} {b:10.3e} {err:10.3e} {4*b:10.3e} {'ok' if ok else 'BAD':>4s}")
                audit.append(dict(case=key,M=M,k=kk,tt=tt,b=b,err=err,ok=bool(ok)))
    print(f"B3 violations: {bad}")

    # full-suite soundness: is 4*B2 >= actual err for EVERY profile?
    print("\n=== full-suite B3 check + certificate table ===")
    u0=cases['logistic w=0.05']
    T=lambda f: min(timeit.repeat(f,number=10,repeat=7))/10
    t_B0=T(lambda: imin(dihiti(u0,x),x)); t_B1=T(lambda: imin(onehilb(u0,x),x))
    print(f"B0 dihiti {t_B0*1e6:.1f}us   B1 onehilb {t_B1*1e6:.1f}us")
    print(f"{'n':>6s} {'k':>2s} {'B3viol':>7s} {'cert%':>7s} {'phys%':>7s} {'adv%':>6s} {'false':>6s} "
          f"{'t_bnd':>7s} {'t_dec':>7s} {'E[t]':>8s} {'vsB0':>6s} {'vsB1':>6s}")
    rows=[]
    for n in ns:
        M=n//2; s=N//n
        tdec=T(lambda s=s: imin(onehilb(u0[::s],x[::s]),x[::s]))
        for kk in ks:
            tbnd=T(lambda kk=kk,M=M: tail_bound(u0,x,M,kk))
            viol=nc=ncp=nca=nf=0; npz=nadv=0
            for key in keys:
                u=cases[key]; D=abs(u[0]-u[-1])
                tr=imin(onehilb(u,x),x); cd=imin(onehilb(u[::s],x[::s]),x[::s])
                b=4*tail_bound(u,x,M,kk)
                if abs(cd-tr) > b*1.001: viol+=1
                cert=abs(cd)>b
                if kind[key]=='phys': npz+=1; ncp+=cert
                else: nadv+=1; nca+=cert
                nc+=cert
                if cert and ((tr>=0)!=(cd>=0)): nf+=1
            tot=npz+nadv; pc=nc/tot; Et=tbnd+tdec+(1-pc)*t_B1
            print(f"{n:6d} {kk:2d} {viol:7d} {100*pc:6.1f}% {100*ncp/npz:6.1f}% {100*nca/max(nadv,1):5.1f}% "
                  f"{nf:6d} {tbnd*1e6:6.1f}u {tdec*1e6:6.1f}u {Et*1e6:7.1f}u {t_B0/Et:6.2f} {t_B1/Et:6.2f}")
            rows.append(dict(n=n,k=kk,viol=viol,cert=pc,cert_phys=ncp/npz,cert_adv=nca/max(nadv,1),
                             false=nf,Et=Et,vsB0=t_B0/Et,vsB1=t_B1/Et))
    json.dump(dict(N=N,audit=audit,rows=rows,t_B0=t_B0,t_B1=t_B1,nkeys=len(keys)),
              open(os.path.join(OUT,f"e1a4_N{N}.json"),"w"),indent=1)

if __name__=="__main__": main()
