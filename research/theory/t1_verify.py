"""Verify the skew counterexample family against the DiHiTI pipeline."""
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
from t1_fast import cl2, m1_arc, PI
import sys, os
sys.path.insert(0,'/Users/kunal/Documents/Spatial KK Project/research/theory')
from t3_clausen import gap_clausen

L=1.0; N=2**16
x=np.linspace(-L,L,N); phi=PI*x/L
TH=np.linspace(-PI,PI,20001)

def build(c,p,q,lam,atom_w=0.004):
    """lam*(narrow arc at c) + (1-lam)*U[p,q], as a density in phi -> eps'(x)."""
    g = lam*((phi>=c-atom_w)&(phi<=c+atom_w))/(2*atom_w) + (1-lam)*((phi>=p)&(phi<=q))/(q-p)
    gt=g*PI/L
    G=cumulative_trapezoid(gt,x,initial=0.0)
    return 1.0+G[-1]-G, g

def closed_form(c,p,q,lam,aw):
    A=lambda th,pp,qq:(cl2(th-qq)-cl2(th-pp))/(qq-pp)
    P=lam*A(TH,c-aw,c+aw)+(1-lam)*A(TH,p,q)
    Pi=lam*A(np.array(PI),c-aw,c+aw)+(1-lam)*A(np.array(PI),p,q)
    return float(P.max()-Pi)

print(f"{'c/pi':>7} {'q/pi':>7} {'lam':>8} {'M1':>10} {'gain(closed)':>13} {'minEps_DiHiTI':>15} {'minEps_L1':>12}")
for cf,qf in [(-0.5,0.90),(-0.5,0.97),(-0.5,0.995),(-0.5,0.999),
              (-0.25,0.995),(-0.75,0.99),(-0.1,0.999),(-0.5,0.9999)]:
    c=cf*PI; q=qf*PI; p=c; aw=0.004
    a=m1_arc(c,aw); b=m1_arc((p+q)/2,(q-p)/2)
    lam=-b/(a-b)
    if not (0<=lam<=1): print(f"{cf:7.2f} {qf:7.4f}  infeasible lam={lam:.3f}"); continue
    e,g=build(c,p,q,lam,aw)
    ei=np.imag(hilbert(np.gradient(e,x))); ei=cumulative_trapezoid(ei,x,initial=0.0)
    ei=ei-np.linspace(ei[0],ei[-1],N)
    dom,_=gap_clausen(e,x)
    m1 = np.trapezoid(g*np.tan(phi/2),phi)
    print(f"{cf:7.2f} {qf:7.4f} {lam:8.4f} {m1:10.2e} {closed_form(c,p,q,lam,aw):13.6f}"
          f" {ei.min():17.6f} {dom.min():13.6f}")
