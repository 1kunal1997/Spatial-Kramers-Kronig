import numpy as np
from bench import onehilb, sign_min_interior, L, A, logistic_u

out = []
def w(s=""):
    print(s)
    out.append(s)

w("## T3(d) sharper adversarial test: perturbation phase-locked to decimation grid\n")
w("Construct a ripple whose period equals the DECIMATED grid spacing and whose phase is zeroed\n")
w("exactly at every decimated sample point, so the decimated grid samples the ripple at its zero\n")
w("crossings and is blind to it, while the N=2^16 ground-truth grid resolves it fully.\n")

N_GT = 2**16
x_gt = np.linspace(-L, L, N_GT, endpoint=True)

w("| n | dx_decim | ripple amp a | ripple T | GT sign | GT interior min | decim sign | decim interior min | decimation MISSES ripple? |")
w("|---|---|---|---|---|---|---|---|---|")

for n in [64, 128, 256]:
    step = N_GT // n
    idx = np.arange(0, N_GT, step)
    x_decim = x_gt[idx]
    dx_decim = x_decim[1] - x_decim[0]
    T = dx_decim  # ripple period = decimated sample spacing
    x0 = x_decim[0]
    for a in [0.05, 0.15, 0.3]:
        def u_adv(x, a=a, T=T, x0=x0):
            return logistic_u(x, 0.05) + a*A*np.sin(2*np.pi*(x-x0)/T)
        u_gt = u_adv(x_gt)
        v_gt = onehilb(u_gt, x_gt)
        s_gt, m_gt = sign_min_interior(v_gt, x_gt)

        u_dec = u_adv(x_decim)
        # check whether decimated samples actually hit the ripple's zero crossings
        ripple_vals_at_samples = a*A*np.sin(2*np.pi*(x_decim-x0)/T)
        v_dec = onehilb(u_dec, x_decim)
        s_dec, m_dec = sign_min_interior(v_dec, x_decim)
        misses = (s_dec != s_gt)
        w(f"| {n} | {dx_decim:.5f} | {a} | {T:.5f} | {int(s_gt)} | {m_gt:.4e} | {int(s_dec)} | {m_dec:.4e} | {misses} |")
        w(f"  (max abs ripple value AT decim samples: {np.max(np.abs(ripple_vals_at_samples)):.2e} -- should be ~0 if phase-locking worked)")

with open("t3d_adv2_output.txt","w") as f:
    f.write("\n".join(out))
print("DONE adv2")
