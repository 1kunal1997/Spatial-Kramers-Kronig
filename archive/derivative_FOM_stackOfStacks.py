# %%

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from plot_functions import plot_setup, plot, legend
import colors

# %% ########################################################
# symbols
N   = sp.symbols('N', positive=True)
'''
T_LR_KK: 0.9808332686181755
R_RL_KK: 0.00030060801840422
A_LR_KK: 0.01916348936694065
A_RL_KK: 0.01886612336341568
alpha:  0.019384558947170817

T0  = 0.9808332686181755
ER  = 0.01886612336341568
EL  = 0.01916348936694065
RRL = 0.00030060801840422
alpha = 0.019384558947170817
'''

T0 = 0.9752423747993312
RRL = 0.000592085801309793
EL = 0.024757527723058424
ER = 0.024165539399367133
alpha = 0.025077547124344565
D1 = 1 - T0
D2 = (1 - T0)**2*(1 + T0)

eN   = sp.exp(-alpha*N)
tN  = T0**N
expr = (ER/D1*(2*sp.log(T0) + 2*alpha) + EL*RRL/D2 * (2*T0*sp.log(T0) + 2*alpha*T0)+
    eN * (ER/D1*(-2*sp.log(T0) - alpha)
        +EL*RRL/D2*(-2*T0*sp.log(T0) - alpha*T0))
  + tN*(ER/D1*(-sp.log(T0)-2*alpha)
        +EL*RRL/D2*(-sp.log(T0)-T0*sp.log(T0)-2*alpha-2*alpha*T0))
  + tN**2*(EL*RRL/D2*(2*alpha))
  + tN**2*eN*(EL*RRL/D2*(-alpha))
  + tN*eN*(ER/D1*(sp.log(T0)+alpha)
         +EL*RRL/D2*(sp.log(T0)+T0*sp.log(T0)+alpha+alpha*T0))
)

# plug in your numbers
vals = {
    T0:   0.9808332686181755,
    ER:   0.01886612336341568,
    EL:   0.01916348936694065,
    RRL:  0.00030060801840422,
    alpha:0.019384558947170817,
}

expr_fixed = sp.simplify(expr.subs(vals))   # expr is your stationary-point LHS
expr_num   = sp.lambdify(N, expr_fixed, "numpy")

Ngrid = np.linspace(1, 2000, 2000)
y = np.real_if_close(expr_num(Ngrid))

xlabel = 'Number of Stacks'; ylabel = 'expr(N)'
title = 'Stationary-point function'
fig,ax = plot_setup(xlabel,ylabel,title=title,figsize=(5,4),auto_scale=True, xlim=(1,2000))
horiz = np.zeros_like(Ngrid)
plot(fig,ax, Ngrid, y , color=colors.blue,auto_scale=True)
plot(fig,ax, Ngrid, horiz, '--', color=colors.black,auto_scale=True)

root = sp.nsolve(expr, 50)   # initial guess N=50
print("N* =", float(root))

# %%
