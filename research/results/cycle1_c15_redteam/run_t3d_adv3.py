import numpy as np
from bench import onehilb, sign_min_interior, L, A, logistic_u

out = []
def w(s=""):
    print(s)
    out.append(s)

w("## T3(d) adversarial test v3: base is a barely-PASSING profile (P1 logistic w=0.05, GT interior min ~1.1e-5)\n")
w("Ripple period = decimated grid spacing, phase-locked to zero at every decimated sample.\n")
w("Goal: find amplitude where GT flips PASS->FAIL but decimated verdict stays PASS (decimation blind to the failure).\n")

N_GT = 2**16
x_gt = np.linspace(-L, L, N_GT, endpoint=True)
u_base = logistic_u(x_gt, 0.05)
v_base = onehilb(u_base, x_gt)
s_base, m_base = sign_min_interior(v_base, x_gt)
w(f"Base profile (no ripple) GT interior min: {m_base:.4e}, sign {int(s_base)}\n")

w("| n | ripple amp a | GT sign | GT interior min | decim sign | decim interior min | DECIMATION FAILS (misses flip to negative)? |")
w("|---|---|---|---|---|---|---|")

found_any = False
for n in [64, 128, 256, 512]:
    step = N_GT // n
    idx = np.arange(0, N_GT, step)
    x_decim = x_gt[idx]
    dx_decim = x_decim[1]-x_decim[0]
    T = dx_decim
    x0 = x_decim[0]
    for a in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        def u_adv(x, a=a, T=T, x0=x0):
            return logistic_u(x, 0.05) + a*A*np.sin(2*np.pi*(x-x0)/T)
        u_gt = u_adv(x_gt)
        v_gt = onehilb(u_gt, x_gt)
        s_gt, m_gt = sign_min_interior(v_gt, x_gt)

        u_dec = u_adv(x_decim)
        v_dec = onehilb(u_dec, x_decim)
        s_dec, m_dec = sign_min_interior(v_dec, x_decim)
        fails = (s_gt < 0 and s_dec > 0)
        if fails:
            found_any = True
        w(f"| {n} | {a} | {int(s_gt)} | {m_gt:.4e} | {int(s_dec)} | {m_dec:.4e} | {fails} |")

w(f"\nAny (n, amplitude) combo where decimation missed a real failure: {found_any}")

with open("t3d_adv3_output.txt","w") as f:
    f.write("\n".join(out))
print("DONE adv3")
