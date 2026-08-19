"""Shared imports/paths for the redteam margin scripts."""
import sys, os
_ROOT = "/Users/kunal/Documents/Spatial KK Project"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'theory'))
sys.path.insert(0, os.path.join(_ROOT, 'theory', 'passivity_2026Aug'))

import numpy as np
import scipy.signal
import scipy.integrate
import tmm_helper as tmm_h
import profile_generator as pg

OUTDIR = os.path.dirname(os.path.abspath(__file__))


def dihiti(u, x):
    du = np.gradient(u, x)
    v = np.imag(scipy.signal.hilbert(du))
    e = scipy.integrate.cumulative_trapezoid(v, x, initial=0.0)
    return e - np.linspace(e[0], e[-1], len(x))


def grid(L, N):
    return np.linspace(-L, L, N)
