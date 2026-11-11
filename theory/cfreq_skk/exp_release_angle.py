"""Item 5 (2026-10-12): what happens AFTER a growing drive stops, and at oblique incidence.

Structure: the B4 design. Horsley Lorentzian sKK target chi = -0.1/(y + 0.15i) x cos^2 window
(|y| <= 2 lam_r), realised with Long et al. Theorem 1 at the GROWING frequency w_c = 2 pi (1 + 0.02i)
using Lorentz layers of quality factor Q = w_r / gam. Analytically this design is EXACTLY a
chirped-resonance medium: w0^2 linear in y (slope -a(2b+gam)/delta), w_p^2 = const x window,
gam = const (Ye et al., Nat. Commun. 8, 51 (2017)). At real w its resonance zone has width
delta_eff = delta gam / (2b + gam), and N layers must resolve it.

Pulse: C-inf turn-on (10 T_r), exponential growth e^{+2 pi 0.02 t} for 60 T_r, C-inf turn-off
(2 T_r). Energies by exact Fourier synthesis (real fields), split into 'prompt' (before the
turn-off starts) and 'delayed' (after). Parseval: the totals equal the |D(w)|^2-weighted
real-frequency R, T, A, whatever the envelope.

Writes data/release_angle.npz (+ a meta JSON string). Run:
  python theory/cfreq_skk/exp_release_angle.py          (~10-15 min; N = 4000 cases dominate)
"""
import sys, os, json, time, subprocess
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import numpy as np
import cfreq_lib as cf

ETA = -0.02
A_AMP, DELTA, YW = -0.1, 0.15, 2.0
DT, NT = 1 / 8, 2 ** 17                    # 16384 T_r window: release of Q = 1e4 fits
T0, T_ON, T_STOP, T_OFF = 5.0, 10.0, 65.0, 2.0
CASES = [(1e4, 400), (1e4, 4000), (1e3, 400), (1e2, 400)]      # (Q, N layers)
THETAS = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 89], float)
TD_ANGLES = [(45.0, 's'), (70.0, 's'), (45.0, 'p')]           # time-domain checks, Q=1e4 N=400


def target(y):
    return cf.lorentzian_profile(y, A_AMP, DELTA) * cf.cos2_window(y, YW)


def design(Q, N, eta=ETA):
    y, chi, d = cf.discretize(target, -YW, YW, N)
    wc = cf.omega_c(eta)
    p = cf.lorentz_design_gamma(chi, wc, wc.real / Q)
    assert p['ok']
    return y, chi, d, wc, p


def budget(Q, N, theta=0.0, pol='s', keep_trace=False):
    y, chi, d, wc, p = design(Q, N)
    t = np.arange(NT) * DT; wr = cf.rfft_omega(NT, DT)
    o = cf.rt_lorentz(p, d, wr, theta=theta, pol=pol)
    oc = cf.rt(cf.eps_lorentz([wc], p), d, [wc], theta=theta, pol=pol)
    dc = cf.pulse_drive(t, ETA, T_ON, T_STOP, T_OFF, T0)
    ds = cf.pulse_drive(t, ETA, T_ON, T_STOP, T_OFF, T0, quadrature=True)
    E = lambda x, m=slice(None): float(np.sum(x[m] ** 2) * DT)
    Ein = E(dc)
    out = dict(pred_rL=abs(oc['rL'][0]), pred_rR=abs(oc['rR'][0]), pred_tau=abs(oc['tauL'][0]))
    for side, (rk, tk) in {'L': ('rL', 'tL'), 'R': ('rR', 'tR')}.items():
        r = cf.synthesize(o[rk], dc); tr = cf.synthesize(o[tk], dc)
        out[f'{side}_R_prompt'] = E(r, t < T_STOP) / Ein
        out[f'{side}_R_delayed'] = E(r, t >= T_STOP) / Ein
        out[f'{side}_T'] = E(tr) / Ein
        out[f'{side}_A'] = 1 - out[f'{side}_R_prompt'] - out[f'{side}_R_delayed'] - out[f'{side}_T']
        if side == 'L':
            rq = cf.synthesize(o[rk], ds)
            m = (t > T_STOP - 20) & (t < T_STOP - 1)
            ratio = np.abs((r - 1j * rq)[m] / (dc - 1j * ds)[m])
            out['L_ratio_drive_min'], out['L_ratio_drive_max'] = float(ratio.min()), float(ratio.max())
            cum = np.cumsum(r[t >= T_STOP] ** 2); cum /= max(cum[-1], 1e-300)
            out['L_release_t50'] = float(t[t >= T_STOP][np.searchsorted(cum, 0.5)] - T_STOP)
            out['L_release_t90'] = float(t[t >= T_STOP][np.searchsorted(cum, 0.9)] - T_STOP)
            if keep_trace:
                out['trace'] = dict(t=t, drive=dc, r=r, tr=tr)
    # Parseval cross-check of the left-incidence totals
    wts = np.abs(np.fft.rfft(dc)) ** 2
    out['L_R_parseval'] = float(np.sum(wts * np.abs(o['rL']) ** 2) / np.sum(wts))
    out['L_T_parseval'] = float(np.sum(wts * np.abs(o['tL']) ** 2) / np.sum(wts))
    return out


def spectra_vs_N(Q=1e4, Ns=(400, 4000, 40000)):
    wc = cf.omega_c(ETA); ww = wc.real * np.linspace(0.85, 1.15, 301)
    res = {}
    for N in Ns:
        y, chi, d, wc, p = design(Q, N)
        o = cf.rt_lorentz(p, d, ww)
        res[N] = (np.abs(o['rL']) ** 2, np.abs(o['rR']) ** 2, np.abs(o['tL']) ** 2)
    return ww / wc.real, res


def angles():
    """R_L, R_R, |tau| vs theta at w_c (ideal target, fine layers) and band-mean real-w values
    of the resolved Q = 100 structure, s and p."""
    wc = cf.omega_c(ETA); a = wc.real
    y, chi, d = cf.discretize(target, -YW, YW, 4000)
    _, _, d4, _, p100 = design(1e2, 400)
    ww = a * np.linspace(0.9, 1.1, 81)
    out = {}
    for pol in ('s', 'p'):
        rows = []
        for th in THETAS:
            oc = cf.rt(1 + chi, d, [wc], theta=th, pol=pol)
            orr = cf.rt(cf.eps_lorentz(ww, p100), d4, ww, theta=th, pol=pol)
            RL = np.abs(orr['rL']) ** 2
            rows.append([abs(oc['rL'][0]) ** 2, abs(oc['rR'][0]) ** 2, abs(oc['tauL'][0]),
                         RL.mean(), (np.abs(orr['rR']) ** 2).mean(),
                         (1 - RL - np.abs(orr['tL']) ** 2).mean()])
        out[pol] = np.array(rows)
    return out


if __name__ == '__main__':
    t_start = time.time()
    data, meta = {}, dict(eta=ETA, A=A_AMP, delta=DELTA, window=YW, dt=DT, nt=NT, t0=T0, t_on=T_ON,
                          t_stop=T_STOP, t_off=T_OFF, cases=CASES, thetas=THETAS.tolist(),
                          td_angles=TD_ANGLES)
    try:
        meta['git'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=_HERE,
                                              text=True).strip()
    except Exception:
        meta['git'] = 'n/a'
    budgets = {}
    for Q, N in CASES:
        b = budget(Q, N, keep_trace=(Q, N) == (1e4, 400))
        if 'trace' in b:
            for k, v in b.pop('trace').items():
                data[f'trace_{k}'] = v
        budgets[f'Q={Q:g},N={N}'] = b
        print(f'Q={Q:g} N={N}: ' + ', '.join(f'{k}={v:.3g}' for k, v in b.items()), flush=True)
    for th, pol in TD_ANGLES:
        b = budget(1e4, 400, theta=th, pol=pol)
        budgets[f'Q=1e4,N=400,th={th:g},{pol}'] = b
        print(f'theta={th:g} {pol}: ' + ', '.join(f'{k}={v:.3g}' for k, v in b.items()), flush=True)
    meta['budgets'] = budgets
    x, sp = spectra_vs_N()
    data['spec_w'] = x
    for N, (RL, RR, T) in sp.items():
        data[f'spec_RL_N{N}'], data[f'spec_RR_N{N}'], data[f'spec_T_N{N}'] = RL, RR, T
    ang = angles()
    data['ang_theta'] = THETAS
    data['ang_s'], data['ang_p'] = ang['s'], ang['p']
    _, _, _, wc, p = design(1e4, 400)
    y, *_ = design(1e4, 400)
    data['design_y'], data['design_w02'], data['design_wp2'] = y, p['w02'], p['wp2']
    os.makedirs(os.path.join(_HERE, 'data'), exist_ok=True)
    np.savez(os.path.join(_HERE, 'data', 'release_angle.npz'), meta=json.dumps(meta), **data)
    print(f'saved data/release_angle.npz ({time.time() - t_start:.0f} s)')
