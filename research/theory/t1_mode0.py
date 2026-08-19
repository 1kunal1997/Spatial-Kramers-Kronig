"""Is 'unimodal about phi_m = 0 (the balance point) + M1=0' already sufficient?"""
import numpy as np
from t1_fast import gain_batch, PI
TH = np.linspace(-PI, PI, 801)
rng = np.random.default_rng(1)
E = 1e-5
best = (-9e9, None)
for _ in range(40):
    K = 30000
    p1 = -np.exp(rng.uniform(np.log(1e-4), np.log(PI-E), K))
    q1 =  np.exp(rng.uniform(np.log(1e-4), np.log(PI-E), K))
    p2 = -np.exp(rng.uniform(np.log(1e-4), np.log(PI-E), K))
    q2 =  np.exp(rng.uniform(np.log(1e-4), np.log(PI-E), K))
    g, lam = gain_batch(p1, q1, p2, q2, TH)
    i = int(np.nanargmax(g))
    if g[i] > best[0]: best = (g[i], (p1[i], q1[i], p2[i], q2[i], lam[i]))
print("phi_m = 0, max gain over 1.2e6 balanced 2-arc extreme points:")
print(f"   max gain = {best[0]:.3e}   at {best[1]}")
print("   (<= 0  =>  PASSIVE for every unimodal-about-0 balanced density)")
