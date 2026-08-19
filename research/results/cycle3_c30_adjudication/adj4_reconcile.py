"""ADJ4: explain the replicator's factor-2.6 and its x0-spread as an edge-probe artifact.
-H[phi](sigma) for both bump shapes: value at the MINIMUM (what Lemma 1 uses) vs at
sigma=-1 / sigma=-1/2 (where an 'offset by delta/2' probe lands)."""
import numpy as np
from scipy.signal import hilbert
M=2**20; s=np.linspace(-60,60,M)
def report(name,phi):
    H=np.imag(hilbert(phi)); g=-H
    i=np.argmax(g); print(f"{name}: max(-H)={g[i]:.5f} at s={s[i]:+.4f}")
    for sig in (-1.0,-0.75,-0.5,-0.25,0.0):
        print(f"    -H(s={sig:+.2f}) = {np.interp(sig,s,g):+.5f}   ratio max/val = {g[i]/max(np.interp(sig,s,g),1e-12):.3f}")
p=np.zeros_like(s); m=np.abs(s)<1; p[m]=np.exp(-1.0/(1.0-s[m]**2))*np.e
report("C^inf compact bump (cycle2 shape)",p)
q=np.zeros_like(s); q[m]=0.5*(1+np.cos(np.pi*s[m]))
report("raised cosine (replicator shape) ",q)
