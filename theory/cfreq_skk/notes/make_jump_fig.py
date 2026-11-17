"""Figure for two_line_conditions.tex, Appendix B: why the single Lorentz recipe is smooth and the
two-line recipe must jump. Same B1s/B8 target (4th-order pole, eta = 0.05)."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cfreq_lib as cf

a = cf.TWO_PI; b = 0.05 * a; wc = a - 1j * b
target = lambda y: 0.5 * 0.2 ** 4 / (y + 0.2j) ** 4 * cf.cos2_window(y, 2.0)
y, chi, d = cf.discretize(target, -2.0, 2.0, 400)
act = np.abs(chi) > 1e-6
p1 = cf.lorentz_design(chi, wc, 0.5 * cf.kappa_max(chi, wc))
p2 = cf.fano2_design(chi, wc)
G = lambda w0, rho=0.36: 1 / (w0 ** 2 - wc ** 2 - 1j * 10 * b * wc) - rho / (w0 ** 2 - wc ** 2 - 1j * 4 * b * wc)
turns = lambda z: np.unwrap(np.angle(z)) / (2 * np.pi)
comp = lambda z: z / np.abs(z).max() * (np.abs(z) / np.abs(z).max()) ** (-0.6)   # radius -> r^0.4

fig, ax = plt.subplots(2, 3, figsize=(10, 6))
# a: what the target needs
ph = turns(chi[act]); ph -= ph[0]
ax[0, 0].plot(y[act], ph, 'k')
ax[0, 0].set_xlabel(r'$y/\lambda_r$'); ax[0, 0].set_ylabel(r'phase of $\chi_t(y)$ (turns)')
ax[0, 0].set_title(f'a  target: phase turns {abs(ph[-1]-ph[0]):.2f} times', loc='left', fontsize=10)
# b: single Lorentz knobs, smooth (gamma left, omega_0 right)
w0s = np.sqrt(p1['w02']) / a; gs = p1['gam'] / (2 * b)
ax[0, 1].plot(y[act], gs[act], 'C3')
ax[0, 1].axhline(1, color='C3', ls=':', lw=0.8)
ax[0, 1].fill_between(y[act], 0, 1, color='C3', alpha=0.08)
ax[0, 1].text(-1.9, 0.12, r'$\gamma<2|\omega^{\prime\prime}|$: pole above the drive line', color='C3', fontsize=8)
ax[0, 1].set_ylim(0, 1.7); ax[0, 1].set_ylabel(r'$\gamma/2|\omega^{\prime\prime}|$', color='C3')
axr = ax[0, 1].twinx(); axr.plot(y[act], w0s[act], 'C0'); axr.set_ylim(0.95, 1.05)
axr.set_ylabel(r'$\omega_0/\omega_r$', color='C0')
ax[0, 1].set_xlabel(r'$y/\lambda_r$'); ax[0, 1].set_title('b  single Lorentz recipe: smooth', loc='left', fontsize=10)
# c: two-line knob, jumps
ax[0, 2].plot(y[act], p2['w0'][act] / a, 'C0.', ms=2)
ax[0, 2].set_xlabel(r'$y/\lambda_r$'); ax[0, 2].set_ylabel(r'$\omega_0/\omega_r$')
ax[0, 2].set_title(r'c  two-line recipe: $\omega_0$ jumps', loc='left', fontsize=10)
# d, e: path of the response as omega_0 sweeps 0 -> infinity (complex plane, radius compressed)
W0 = a * np.concatenate([np.linspace(1e-3, 0.7, 400), np.linspace(0.7, 1.3, 4000), np.linspace(1.3, 60, 4000)])
one = 1 / (W0 ** 2 - wc ** 2 - 1j * 10 * b * wc)
for axx, z, ttl in ((ax[1, 0], one, r'd  ONE line, fixed $\gamma$: $1/D(\omega_c)$'),
                    (ax[1, 1], G(W0), r'e  two lines: $G(\omega_0)$')):
    zc = comp(z)
    axx.plot(zc.real, zc.imag, 'C0')
    for w, lab in ((W0[0], r'$\omega_0\to0$'), (a, r'$\omega_0=\omega_r$'), (W0[-1], r'$\omega_0\to\infty$')):
        k = np.argmin(abs(W0 - w)); axx.plot(zc[k].real, zc[k].imag, 'ko', ms=4)
        axx.annotate(lab, (zc[k].real, zc[k].imag), textcoords='offset points', xytext=(4, 4), fontsize=8)
    axx.axhline(0, color='gray', lw=0.6); axx.axvline(0, color='gray', lw=0.6)
    axx.set_aspect('equal'); axx.set_xlim(-1.15, 1.15); axx.set_ylim(-1.15, 1.15)
    axx.text(0.03, 0.95, 'loss (Im > 0)', transform=axx.transAxes, fontsize=7, va='top', color='gray')
    axx.text(0.03, 0.03, 'gain (Im < 0)', transform=axx.transAxes, fontsize=7, color='gray')
    axx.set_xlabel('Re (radius compressed)'); axx.set_ylabel('Im')
    axx.set_title(ttl, loc='left', fontsize=10)
# f: how far each can turn vs what the target needs
Wf = a * np.logspace(-3, 2, 20001)
t1 = turns(1 / (Wf ** 2 - wc ** 2 - 1j * 10 * b * wc)); t2 = turns(G(Wf))
ax[1, 2].semilogx(Wf / a, t1[0] - t1, 'C1', label=f'one line: {abs(t1[-1]-t1[0]):.2f} turns')
ax[1, 2].semilogx(Wf / a, t2[0] - t2, 'C0', label=f'two lines: {abs(t2[-1]-t2[0]):.2f} turns')
ax[1, 2].axhline(abs(ph[-1] - ph[0]), color='k', ls='--', lw=0.8, label=f'target needs {abs(ph[-1]-ph[0]):.2f}')
ax[1, 2].set_xlabel(r'$\omega_0/\omega_r$'); ax[1, 2].set_ylabel('phase turned (turns)')
ax[1, 2].legend(fontsize=8, loc='upper left'); ax[1, 2].set_ylim(-0.05, 2.3); ax[1, 2].set_title(r'f  phase reachable with $\omega_0$ alone', loc='left', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'two_line_jumps.png'), dpi=200)
print('one line', t1[-1] - t1[0], 'two lines', t2[-1] - t2[0], 'target', ph[-1] - ph[0])
print('two-line arg at w0->0, w_r, inf (turns):', np.angle(G(Wf[0]))/(2*np.pi), np.angle(G(a))/(2*np.pi), np.angle(G(Wf[-1]))/(2*np.pi))
