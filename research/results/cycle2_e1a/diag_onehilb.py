"""Does the 'best baseline' onehilb agree with dihiti on the LABEL? And is the
decimated-logistic MISS a decimation error or a transform-convention bias?"""
import numpy as np
from e1a import dihiti, onehilb, interior_min, L, A, decim
for w in [0.3,0.15,0.05,0.02,0.005]:
    print(f"--- logistic w={w}")
    for N in [2**11,2**12,2**13,2**14,2**16,2**18]:
        x=np.linspace(-L,L,N); u=1+A/(1+np.exp(x/w)); D=u[0]-u[-1]
        d=interior_min(dihiti(u,x),x)/D; o=interior_min(onehilb(u,x),x)/D
        print(f"  N={N:7d} dihiti {d:+.4e}  onehilb {o:+.4e}  diff {o-d:+.3e}")
