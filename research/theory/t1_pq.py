"""T1: worst-case gain over {mu unimodal, M1=0, supp in [P,Q]} via its 2-arc extreme points."""
import numpy as np, sys
from scipy.optimize import differential_evolution
from t1_fast import cl2, phi_arc, m1_arc, PI
TH = np.linspace(-PI, PI, 801)

def gain_arcs(p1,q1,p2,q2):
    """balanced mixture of U[p1,q1], U[p2,q2]; requires overlap for unimodality."""
    if q1-p1 < 1e-7 or q2-p2 < 1e-7: return -9.0
    if min(q1,q2) < max(p1,p2) - 1e-12: return -9.0          # disjoint -> bimodal
    a = m1_arc((p1+q1)/2,(q1-p1)/2); b = m1_arc((p2+q2)/2,(q2-p2)/2)
    if abs(a-b) < 1e-14: return -9.0
    lam = -b/(a-b)
    if not (0.0 <= lam <= 1.0): return -9.0
    P  = lam*phi_arc(TH,(p1+q1)/2,(q1-p1)/2) + (1-lam)*phi_arc(TH,(p2+q2)/2,(q2-p2)/2)
    Pi = lam*phi_arc(np.array(PI),(p1+q1)/2,(q1-p1)/2) + (1-lam)*phi_arc(np.array(PI),(p2+q2)/2,(q2-p2)/2)
    return float(P.max()-Pi)

def worst(P,Q,seeds=(0,1)):
    span = Q-P
    def neg(u):
        p1 = P+u[0]*span; q1 = p1+u[1]*(Q-p1)
        p2 = P+u[2]*span; q2 = p2+u[3]*(Q-p2)
        return -gain_arcs(p1,q1,p2,q2)
    best=(-9e9,None)
    for s in seeds:
        r = differential_evolution(neg,[(0,1)]*4,seed=s,maxiter=120,popsize=18,tol=1e-13,polish=True)
        if -r.fun>best[0]:
            u=r.x; p1=P+u[0]*span; q1=p1+u[1]*(Q-p1); p2=P+u[2]*span; q2=p2+u[3]*(Q-p2)
            best=(-r.fun,(p1,q1,p2,q2))
    return best

if __name__=='__main__':
    print("worst gain over unimodal balanced mu with supp = [P,Q].  (>0 => FAIL exists)")
    print(f"{'P/pi':>7} {'Q/pi':>7} {'(Q-P)/pi':>9} {'maxgain':>11}   worst arcs/pi")
    grid=[(-0.4,0.4),(-0.45,0.45),(-0.49,0.49),(-0.3,0.5),(-0.2,0.6),(-0.1,0.7),
          (-0.2,0.75),(-0.1,0.8),(-0.05,0.85),(-0.3,0.65),(-0.4,0.55),(-0.45,0.5),
          (-0.6,0.9),(-0.8,0.95),(-0.9,0.9),(-0.95,0.95),(-0.5,0.95),(-0.2,0.95)]
    for (Pp,Qp) in grid:
        P,Q=Pp*PI,Qp*PI
        g,c=worst(P,Q)
        print(f"{Pp:7.3f} {Qp:7.3f} {Qp-Pp:9.3f} {g:11.6f}   "
              f"({c[0]/PI:+.3f},{c[1]/PI:+.3f} | {c[2]/PI:+.3f},{c[3]/PI:+.3f})")
