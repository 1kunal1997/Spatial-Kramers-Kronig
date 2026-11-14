"""B3a check: COMSOL (built-in temw + Drude-Lorentz, build_b3a.py) vs our TMM x FFT synthesis.

  a  the B3a quantity |t(t)/t_free(t)| / |tau(w_c)| for the virtual-gain slab (red,
     gamma/2|w''| = 0.5) and the lossy slab (blue, 1.5). Lines: TMM x FFT (exactly as in
     fig_cfreq_feasibility.fig_time). Circles: COMSOL.
  b  agreement: |E_COMSOL - E_TMMxFFT| / |E_TMMxFFT| for the complex exit-face field, for each
     available COMSOL time step (convergence check).
Run: python comsol/fig_b3a_comsol.py   (after build_b3a.py)
"""
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_CF = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_CF))
sys.path[:0] = [_ROOT, _CF]
import numpy as np
import theory.skk_fig_common as C
import cfreq_lib as cf
from fig_cfreq_feasibility import DT, NT, T_ON, T0

C.apply_style()
DATA = os.path.join(_CF, 'data')


def tmm_fft(chi_im, ratio, eta, Ls=1.0, nl=40):
    """Complex exit-face field (cos - i sin drives) by TMM x FFT, and tau(w_c)."""
    wc = cf.omega_c(eta); b = abs(wc.imag)
    p = cf.lorentz_design_gamma(np.array([1j * chi_im]), wc, ratio * 2 * b)
    d = np.full(nl, Ls / nl)
    e = lambda w: np.repeat(cf.eps_lorentz(w, p), nl, axis=1)
    tau_c = cf.rt(e([wc]), d, [wc])['tauL'][0]
    wr = cf.rfft_omega(NT, DT)
    t = np.arange(NT) * DT
    _, out = cf.complex_response(cf.rt(e(wr), d, wr)['tL'], t, eta, T_ON, T0)
    return t, out, tau_c


def drive_c(t, eta):
    """Complex drive at the entry face (cos - i sin), same turn-on as the synthesis."""
    return (cf.decaying_drive(t, eta, T_ON, T0)
            - 1j * cf.decaying_drive(t, eta, T_ON, T0, quadrature=True))


def main():
    files = sorted(glob.glob(os.path.join(DATA, 'b3a_comsol_dt*.npz')))
    runs = {os.path.basename(f)[11:-4]: np.load(f) for f in files}
    ref = runs.get('dt100_rho0.75') or next(iter(runs.values()))
    eta, Ls = float(ref['eta']), float(ref['L'])
    combos = ref['combos']

    fig, axs, _ = C.panel_grid(1, 2, margin_l=0.47, col_gap=0.65); ax_a, ax_b = axs[0]
    styles = {'dt100_rho0.75': ('-', r'COMSOL step $T_r/100$'), 'dt200_rho0.75': ('--', r'COMSOL step $T_r/200$'),
              'dt400_rho0.75': (':', r'COMSOL step $T_r/400$')}
    for k in range(0, len(combos), 2):
        chi_im, ratio = combos[k, 0], combos[k, 1]
        col = C.C_EPS_IM if chi_im < 0 else C.C_EPS_RE
        t, out, tau_c = tmm_fft(chi_im, ratio, eta, Ls)
        free = drive_c(t - Ls, eta)                       # free propagation to the exit face
        ok = np.abs(free) > 1e-10 * np.abs(free).max()
        q = np.where(ok, np.abs(out) / np.where(ok, np.abs(free), 1), np.nan)
        tt = t - T0 - Ls                                  # time after turn-on at the exit face
        lab = (rf'$\epsilon(\omega_c)=1{chi_im:+.1f}i$, $\gamma/2|\omega^{{\prime\prime}}|={ratio:g}$'
               ' (TMM$\\times$FFT)')
        ax_a.semilogy(tt, q, color=col, lw=C.LW_THIN, label=lab)
        ax_a.axhline(abs(tau_c), color=col, lw=C.LW_GUIDE, ls=':')

        for name, R in runs.items():
            tc = R['t']; ec = R['e_exit'][k] - 1j * R['e_exit'][k + 1]
            fc = drive_c(tc - Ls, eta)
            okc = np.abs(fc) > 1e-10
            qc = np.abs(ec[okc]) / np.abs(fc[okc])
            lsty, mlab = styles.get(name, ('-.', name))
            if name == 'dt100_rho0.75':
                tq = tc[okc] - T0 - Ls
                sel = (tq >= 1) & (np.abs(np.round(tq) - tq) < 1e-6)   # one circle per period
                ax_a.semilogy(tq[sel], qc[sel], ls='none', marker='o', ms=2.6,
                              mfc='none', mec=col, mew=0.6)
            # b: complex-field agreement on the COMSOL time grid
            eref = np.interp(tc, t, out.real) + 1j * np.interp(tc, t, out.imag)
            rel = np.abs(ec - eref) / np.abs(eref)
            m = (tc - T0 - Ls >= 1) & okc
            ax_b.semilogy(tc[m] - T0 - Ls, rel[m], color=col, lw=C.LW_THIN, ls=lsty)
            print(f'{name}  chi_im={chi_im:+.1f} ratio={ratio:g}: median rel. diff '
                  f'{np.nanmedian(rel[m]):.1e}, max {np.nanmax(rel[m]):.1e};  '
                  f'B3a quantity at t=60: COMSOL {np.interp(60, tc[okc]-T0-Ls, qc):.3g}')
        print(f'   TMMxFFT B3a quantity at t=60: {np.interp(60, tt[ok], q[ok]):.3g}')

    ax_a.plot([], [], ls='none', marker='o', ms=2.6, mfc='none', mec='k', mew=0.6,
              label='COMSOL (temw, built-in Drude–Lorentz)')
    ax_a.set_xlim(0, 60); ax_a.set_ylim(1e-2, 1e3)
    ax_a.set_xlabel(r'time after turn-on ($T_r$)')
    ax_a.set_ylabel(r'$|t(t)/t_{\rm free}(t)|$')
    ax_a.legend(loc='upper left', fontsize=5)

    for name in runs:
        ax_b.plot([], [], color='k', lw=C.LW_THIN, ls=styles.get(name, ('-.', name))[0],
                  label=styles.get(name, ('', name))[1])
    ax_b.set_xlim(0, 60)
    ax_b.set_xlabel(r'time after turn-on ($T_r$)')
    ax_b.set_ylabel(r'rel. difference of complex $E_{\rm exit}$')
    ax_b.set_ylim(1e-5, 1); ax_b.legend(loc='lower right', fontsize=5)
    C.label_panels(fig, [(ax_a, 'a'), (ax_b, 'b')])
    out = os.path.join(_CF, 'figures', 'B3a_comsol_check.png')
    C.save(fig, out); print('saved', out)


if __name__ == '__main__':
    main()
