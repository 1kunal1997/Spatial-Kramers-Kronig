"""Validate the circle/arc form against DiHiTI on profiles built from arc mixtures."""
import sys, os
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
from t1_arcs import cl2, phi_arc, m1_arc

def dihiti(e, x):
    v = np.imag(hilbert(np.gradient(e, x)))
    c = cumulative_trapezoid(v, x, initial=0.0)
    return c - np.linspace(c[0], c[-1], len(x))

L = 1.0; N = 2**15
x = np.linspace(-L, L, N); phi = np.pi*x/L

def build(arcs):
    """arcs = [(lam, c, w)]  ->  eps'(x) with drop density = sum lam*Unif[c-w,c+w] (in phi),
       total drop D = 1."""
    g = np.zeros_like(phi)                      # density in phi
    for lam, c, w in arcs:
        g += lam*((phi >= c-w) & (phi <= c+w))/(2*w)
    gt = g*np.pi/L                              # density in x
    G = cumulative_trapezoid(gt, x, initial=0.0)
    return 1.0 + G[-1] - G, g

def predict(arcs, theta):
    P  = sum(lam*phi_arc(theta, c, w) for lam, c, w in arcs)
    Pi = sum(lam*phi_arc(np.pi,  c, w) for lam, c, w in arcs)
    return (Pi - P)/np.pi

tests = [ [(1.0, 0.0, 1.2)],
          [(0.5, -0.8, 0.5), (0.5, 0.9, 0.4)],
          [(0.3, 0.4, 0.3), (0.7, 0.4, 2.2)],
          [(0.6, 1.5, 1.0), (0.4, -1.9, 0.9)] ]
print("circle/arc closed form vs DiHiTI:")
for arcs in tests:
    e, g = build(arcs)
    ei = dihiti(e, x)
    pr = predict(arcs, phi)
    m = np.abs(x) < 0.999*L
    print(f"  arcs={arcs}")
    print(f"     max|pred-DiHiTI| = {np.abs(pr-ei)[m].max():.3e}   scale {ei.max():.4f}"
          f"   M1 = {sum(lam*m1_arc(c,w) for lam,c,w in arcs):+.4f}")
