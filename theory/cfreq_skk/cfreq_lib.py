"""cfreq_lib — spatial Kramers-Kronig (sKK) profiles at COMPLEX frequency.

Idea B (2026-10-12): Long, Catrysse, Han & Fan, Light Sci. Appl. 15:48 (2026), Theorem 1 —
at a complex frequency w_c = a - i|b| (a temporally DECAYING excitation, e^{-i w t}
convention) a single-pole PASSIVE Lorentz medium can be given any complex permittivity,
including Im eps < 0. So sKK profiles that need gain at real frequency (e.g. the King,
Horsley & Philbin PRL 118, 163201 (2017) unit-transmission profiles, double poles in the
lower half position plane) can be built point by point from passive materials.

Units: c = 1, lengths in units of the real-part wavelength lam_r = 2 pi c / a  =>  a = 2 pi.
The decay ratio  eta = |b| / a  is the one excitation knob: w_c = 2 pi (1 - i eta).

Conventions: time dependence e^{-i w t}; passive <=> Im eps(w) > 0 for real w > 0;
Lorentz eps(w) = 1 + wp2 / (w02 - w^2 - i gam w), with wp2, w02, gam > 0.
Background on both sides is vacuum (eps = 1). s-polarisation, real transverse wavevector kx.
"""
import numpy as np

TWO_PI = 2 * np.pi


def omega_c(eta):
    """Complex excitation frequency for decay ratio eta (eta = 0 -> real frequency;
    eta < 0 -> exponentially GROWING drive)."""
    return TWO_PI * (1 - 1j * eta)


# ---------------------------------------------------------------------------
# Target profiles chi(y) = eps(y) - 1, analytic in the UPPER half y-plane
# ---------------------------------------------------------------------------
def pole_profile(y, poles, amps, order):
    """chi(y) = sum_k a_k / (y - z_k)^order, Im z_k < 0. order >= 2 -> int chi dy = 0 ->
    unit transmission (KHP 2017 'cancellation condition'); higher order -> faster tails."""
    y = np.asarray(y, dtype=float)[..., None]
    return np.sum(np.asarray(amps) / (y - np.asarray(poles)) ** order, axis=-1)


def cos2_window(y, Y, frac=0.5):
    """1 on |y| <= (1-frac) Y, cos^2 roll-off to 0 at |y| = Y (smooth tail truncation)."""
    u = np.clip((np.abs(y) - (1 - frac) * Y) / (frac * Y), 0, 1)
    return np.cos(0.5 * np.pi * u) ** 2


def khp_profile(y, poles, amps):
    """King-Horsley-Philbin perfect-transmission profile: sum of DOUBLE poles
    chi(y) = sum_k a_k / (y - z_k)^2,  Im z_k < 0.  R = 0 from the left and t = 1 exactly
    (infinite profile, real frequency); Im chi changes sign -> needs gain at real w."""
    y = np.asarray(y, dtype=float)[..., None]
    return np.sum(np.asarray(amps) / (y - np.asarray(poles)) ** 2, axis=-1)


def lorentzian_profile(y, A, delta, y0=0.0):
    """Horsley single simple pole chi = A / (y - y0 + i delta): sKK absorber (A < 0 passive)."""
    return A / (np.asarray(y) - y0 + 1j * delta)


def random_khp(n_poles, x_span, delta_range, chi_max, seed=0, order=2):
    """Disordered KHP profile: n_poles double poles at random x in [-x_span, x_span],
    depth delta in delta_range, random complex amplitudes; rescaled so max|chi| = chi_max on a
    fine grid. Returns (poles, amps)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-x_span, x_span, n_poles)
    d = rng.uniform(*delta_range, n_poles)
    poles = x - 1j * d
    amps = (rng.normal(size=n_poles) + 1j * rng.normal(size=n_poles)) * d ** order
    yy = np.linspace(-x_span - 5 * delta_range[1], x_span + 5 * delta_range[1], 20001)
    amps *= chi_max / np.abs(pole_profile(yy, poles, amps, order)).max()
    return poles, amps


# ---------------------------------------------------------------------------
# Long et al. Theorem 1, closed form: passive Lorentz parameters hitting eps(w_c) = 1 + chi
# ---------------------------------------------------------------------------
def lorentz_design(chi, wc, kappa):
    """Per-point passive Lorentz parameters with eps(wc) = 1 + chi exactly.

    With wc = a + i b (b < 0), D = w02 - wc^2 - i gam wc = X + i a (2|b| - gam), and
    wp2 = chi * D must be real positive, i.e. D = lam * conj(chi), lam > 0. Choosing
    gam = 2|b| + kappa * Im(chi) makes lam = a * kappa (one knob for the whole profile):
        wp2 = a kappa |chi|^2,   gam = 2|b| + kappa Im chi,
        w02 = a kappa Re chi + a^2 - b^2 + gam |b|.
    Gain points (Im chi < 0) get gam < 2|b|: resonators that ring LONGER than the excitation
    decays (Long et al. SI Eq. S18). Feasibility needs gam > 0 and w02 > 0 (checked).
    Returns dict(wp2, w02, gam, ok)."""
    a, b = wc.real, wc.imag
    chi = np.asarray(chi, dtype=complex)
    wp2 = a * kappa * np.abs(chi) ** 2
    gam = 2 * abs(b) + kappa * chi.imag
    w02 = a * kappa * chi.real + a ** 2 - b ** 2 + gam * abs(b)
    ok = bool(np.all(gam > 0) and np.all(w02 > 0))
    return dict(wp2=wp2, w02=w02, gam=gam, ok=ok)


def lorentz_design_gamma(chi, wc, gam):
    """Same target eps(wc) = 1 + chi, with the damping gam chosen per point (the free
    parameter of Long et al.'s one-parameter family). wc = a + i b with b of EITHER sign:
    D = w02 - wc^2 - i gam wc must equal lam conj(chi), lam = a (2b + gam) / Im chi > 0.
      decaying drive (b < 0): Im chi < 0 needs gam < 2|b| -> the material pole -i gam/2 lies
        BETWEEN the real axis and wc; co-decaying with the drive the oscillator has damping
        gam - 2|b| < 0 (unstable): the w_c steady state is not reachable (Lalanne & Wu 2025).
      growing drive (b > 0): only Im chi > 0 is reachable, but even gam -> 0 (LOSSLESS media)
        works, and every pole lies below wc, so the steady state is always reached."""
    a, b = wc.real, wc.imag
    chi = np.asarray(chi, dtype=complex)
    gam = np.broadcast_to(np.asarray(gam, float), chi.shape).copy()
    lam = a * (2 * b + gam) / chi.imag
    wp2 = lam * np.abs(chi) ** 2
    w02 = lam * chi.real + a ** 2 - b ** 2 - gam * b
    ok = bool(np.all(gam >= 0) and np.all(w02 > 0) and np.all(lam > 0))
    return dict(wp2=wp2, w02=w02, gam=gam, ok=ok)


def kappa_max(chi, wc):
    """Largest kappa with gam > 0 everywhere (gam = 2|b| + kappa Im chi)."""
    neg = -np.min(np.imag(chi))
    return np.inf if neg <= 0 else 2 * abs(wc.imag) / neg


def eps_lorentz(w, p):
    """eps(w) for every layer: shape (len(w), n_layers). w may be complex."""
    w = np.asarray(w, dtype=complex)[:, None]
    return 1 + p['wp2'][None, :] / (p['w02'][None, :] - w ** 2 - 1j * p['gam'][None, :] * w)


# ---------------------------------------------------------------------------
# s-pol transfer matrix (E, dE/dy), vectorised over frequency; complex w allowed
# ---------------------------------------------------------------------------
def _layer_product(eps, d, k0, kx, mu=None, pol='s'):
    """Product of characteristic matrices, layers ordered left -> right, for the tangential
    pair (F, F'/Z): s-pol F = E_z, Z = mu; p-pol F = H_z, Z = eps (mu = 1).
    eps, mu: (nw, N) complex (mu=None -> 1), d: (N,), k0, kx: (nw,) or scalars. The layer
    matrix is even in q, so no branch choice is needed inside the slab."""
    nw = eps.shape[0]
    if mu is None:
        mu = np.ones_like(eps)
    imp = mu if pol == 's' else eps
    m11 = np.ones(nw, complex); m12 = np.zeros(nw, complex)
    m21 = np.zeros(nw, complex); m22 = np.ones(nw, complex)
    for j in range(eps.shape[1]):
        q2 = k0 ** 2 * eps[:, j] * mu[:, j] - kx ** 2
        q = np.sqrt(q2)
        c = np.cos(q * d[j])
        s_over_q = imp[:, j] * d[j] * np.sinc(q * d[j] / np.pi)   # Z sin(q d)/q
        qs = -q2 * d[j] * np.sinc(q * d[j] / np.pi) / imp[:, j]   # -q sin(q d)/Z
        # new = L_j @ old
        m11, m12, m21, m22 = (c * m11 + s_over_q * m21, c * m12 + s_over_q * m22,
                              qs * m11 + c * m21, qs * m12 + c * m22)
    return m11, m12, m21, m22


def _rt_from_M(M, kb):
    m11, m12, m21, m22 = M
    A = 1j * kb * m11 - m21
    B = -kb ** 2 * m12 - 1j * kb * m22
    r = -(A + B) / (A - B)
    t = m11 * (1 + r) + m12 * 1j * kb * (1 - r)
    return r, t


def rt(eps, d, w, kx=0.0, mu=None, pol='s', theta=None):
    """Reflection/transmission amplitudes for incidence from the LEFT and from the RIGHT.

    kx: scalar or (nw,) transverse wavevector. theta (deg, in vacuum) overrides it with
    kx = w sin(theta) at EVERY frequency, complex ones included: a pulse arriving from a
    fixed direction, field ~ f(t - x sin(theta) - y cos(theta)). pol 's' (E_z) or 'p' (H_z).

    eps (nw, N) or (N,) layer permittivities at the frequencies w (nw,), thicknesses d (N,),
    vacuum on both sides. Returns dict(rL, tL, rR, tR, tau) where t is the exit-face field /
    entry-face incident field, and tau = t * exp(-i kb L) is the transmission RELATIVE to free
    propagation over the same length (tau = 1 <=> the slab is invisible)."""
    w = np.atleast_1d(np.asarray(w, dtype=complex))
    w = np.where(w == 0, 1e-9, w)              # DC: kb = 0 makes r singular; nudge it
    eps = np.atleast_2d(eps)
    if eps.shape[0] != w.size:
        eps = np.broadcast_to(eps, (w.size, eps.shape[-1]))
    if mu is not None:
        mu = np.atleast_2d(mu)
        if mu.shape[0] != w.size:
            mu = np.broadcast_to(mu, (w.size, mu.shape[-1]))
    k0 = w                                     # c = 1
    if theta is not None:
        kx = w * np.sin(np.radians(theta))
    kb = np.sqrt(k0 ** 2 - kx ** 2)
    kb = np.where(kb.real < 0, -kb, kb)        # forward branch, continuous from real w
    d = np.asarray(d, float)
    rL, tL = _rt_from_M(_layer_product(eps, d, k0, kx, mu, pol), kb)
    rR, tR = _rt_from_M(_layer_product(eps[:, ::-1], d[::-1], k0, kx,
                                       None if mu is None else mu[:, ::-1], pol), kb)
    L = d.sum()
    return dict(rL=rL, tL=tL, rR=rR, tR=tR, tauL=tL * np.exp(-1j * kb * L),
                tauR=tR * np.exp(-1j * kb * L), kb=kb)


def discretize(fun, y_lo, y_hi, n):
    """Uniform n-layer staircase of chi = fun(y) on [y_lo, y_hi] (midpoint values)."""
    edges = np.linspace(y_lo, y_hi, n + 1)
    ym = 0.5 * (edges[1:] + edges[:-1])
    return ym, fun(ym), np.diff(edges)


# ---------------------------------------------------------------------------
# Time domain by Fourier synthesis (exact for the linear, time-invariant passive structure)
# ---------------------------------------------------------------------------
def _ramp(tt, t_on, kind):
    """Turn-on envelope, 0 for tt <= 0 and 1 for tt >= t_on. 'sin2' is C1 at the ends;
    'planck' is C-infinity (all derivatives vanish at both ends) -> adiabatic turn-on."""
    u = np.clip(tt / max(t_on, 1e-12), 0, 1)
    if kind == 'sin2':
        return np.sin(0.5 * np.pi * u) ** 2
    out = np.zeros_like(u)
    out[u >= 1] = 1.0
    m = (u > 0) & (u < 1)
    z = 1 / u[m] - 1 / (1 - u[m])
    out[m] = 0.5 * (1 - np.tanh(z / 2))              # = 1/(1+e^z), overflow-safe
    return out


def decaying_drive(t, eta, t_on, t0=0.0, quadrature=False, ramp='planck'):
    """Real incident field at the entry face: smooth sin^2 turn-on over t_on, then
    e^{-|b| t} cos(a t) (= Re e^{-i w_c t}); sin(a t) if quadrature. Zero before t0.
    Driving with cos and sin separately and combining out_c = out_cos - i out_sin gives the
    exact complex-frequency output (|out_c| = |R(w_c)| |drive_c| in steady state) without
    Hilbert-transform envelope artefacts."""
    a, b = TWO_PI, TWO_PI * eta
    tt = t - t0
    osc = np.sin(a * tt) if quadrature else np.cos(a * tt)
    return _ramp(tt, t_on, ramp) * np.exp(-b * np.clip(tt, 0, None)) * osc


def complex_response(resp, t, eta, t_on, t0=0.0, ramp='planck'):
    """(drive_c, out_c): complex drive ramp*e^{-i w_c t} and the exact output of `resp`."""
    dc = decaying_drive(t, eta, t_on, t0, ramp=ramp)
    ds = decaying_drive(t, eta, t_on, t0, quadrature=True, ramp=ramp)
    return dc - 1j * ds, synthesize(resp, dc) - 1j * synthesize(resp, ds)


def pulse_drive(t, eta, t_on, t_stop, t_off, t0=0.0, quadrature=False):
    """Finite pulse: C-inf turn-on over t_on starting at t0, exponential envelope e^{-b t}
    (b = 2 pi eta; eta < 0 GROWS), C-inf turn-off over t_off starting at t_stop."""
    a, b = TWO_PI, TWO_PI * eta
    tt = t - t0
    env = _ramp(tt, t_on, 'planck') * (1 - _ramp(t - t_stop, t_off, 'planck'))
    osc = np.sin(a * tt) if quadrature else np.cos(a * tt)
    return env * np.exp(-b * np.clip(tt, 0, t_stop + t_off - t0)) * osc


def rfft_omega(n, dt):
    """Angular frequencies of np.fft.rfft for n samples spaced dt."""
    return TWO_PI * np.fft.rfftfreq(n, dt)


def synthesize(resp, drive):
    """Output signal for a real drive through a response resp(w) given in the PHYSICS
    convention e^{-i w t}, sampled at rfft_omega(len(drive), dt). numpy's inverse FFT builds
    signals from e^{+i w t}, so the response is applied as conj(resp)."""
    return np.fft.irfft(np.fft.rfft(drive) * np.conj(resp), n=drive.size)


def complex_response_shifted(resp_fn, t, eta, t_on, t0=0.0, c=0.0, ramp='planck'):
    """complex_response with the Fourier contour moved to Im w = -c (0 <= c < every pole's
    |Im w_p| and < |b|): out(t) e^{ct} = IFT[ resp(nu - ic) FT[drive e^{ct}] ]. Exact (no pole
    is crossed) and it removes the e^{-ct} dynamic-range loss of the decaying drive, so the
    output/drive RATIO can be followed far longer than with c = 0 in double precision.
    resp_fn(w) must accept complex w. Returns (drive_c e^{ct}, out_c e^{ct}); ratio unchanged."""
    dt = t[1] - t[0]
    w = rfft_omega(t.size, dt) - 1j * c
    resp = resp_fn(w)
    eta_eff = eta - c / TWO_PI                   # drive * e^{c(t-t0)}, built without overflow
    dc = decaying_drive(t, eta_eff, t_on, t0, ramp=ramp)
    ds = decaying_drive(t, eta_eff, t_on, t0, quadrature=True, ramp=ramp)
    # physics e^{-i w t}; numpy irfft uses e^{+i nu t} -> apply conj(resp(-nu* ...)): for a real
    # system resp(-nu - ic) = conj(resp(nu - ic)), so conj(resp) on nu >= 0 is the right kernel
    out = lambda x: np.fft.irfft(np.fft.rfft(x) * np.conj(resp), n=x.size)
    return dc - 1j * ds, out(dc) - 1j * out(ds)


def rt_lorentz(p, d, w, chunk=4096, **kw):
    """rt() for a Lorentz-layer structure, evaluating eps(w) in frequency chunks so that
    (n_w x N) never has to sit in memory at once (N = 4000 layers x 65k frequencies = 4 GB)."""
    w = np.atleast_1d(np.asarray(w, dtype=complex))
    kx = kw.pop('kx', 0.0)
    parts = []
    for i in range(0, w.size, chunk):
        ws = w[i:i + chunk]
        kxs = kx[i:i + chunk] if np.ndim(kx) else kx
        parts.append(rt(eps_lorentz(ws, p), d, ws, kx=kxs, **kw))
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}


# ---------------------------------------------------------------------------
# STABLE virtual gain: two-pole passive media (2026-10-12, B7)
# ---------------------------------------------------------------------------
# A single Lorentz term has Im chi(w_c) < 0 only if gam < 2|b|: its pole rings longer than the
# drive decays, so the w_c steady state is never reached (B3). Two-pole media evade this: every
# pole can decay FASTER than the drive (margin M = slowest pole decay / |b| > 1) while
# Im chi(w_c) < 0, because the drive's decay re-weights the two poles differently.
#   'eit'  : textbook EIT. Lossy bright oscillator x1 (driven, carries P) stiffness-coupled to a
#            LOSSLESS dark oscillator x2. Passive by construction (any g1, g2 >= 0). In the frame
#            co-decaying with the drive the dark mode has gain (g2 - 2|b| < 0) and the bright mode
#            loss; strong coupling shares the loss, so every hybrid mode is still damped.
#   'fano2': broad Lorentz line minus a narrow one at the same w0 (an EIT-like hole), f1 > 0,
#            f2 >= 0. Passive at every real w  <=>  f2/f1 <= g2/g1 (exact, see fano2_design).

def eit_shape(wc, g1=10.0, g2=0.0, kappa=6.0, det=-0.2):
    """Global EIT shape in units of |b| = |Im w_c| (so it scales with eta): bright damping g1|b|,
    dark damping g2|b|, coupling kappa a|b| (units of w^2), dark-mode detuning det|b| (dark
    resonance at w0 + det|b|). Defaults: chosen by a grid search so that EVERY phase of
    chi(w_c) is reachable with slowest-mode decay >= 2|b| and the least oscillator strength."""
    a, b = wc.real, abs(wc.imag)
    return dict(g1=g1 * b, g2=g2 * b, kappa=kappa * a * b, det=det * b)


def _eit_g(w, w0, s):
    """chi / wp2 of the EIT medium: D2 / (D1 D2 - kappa^2)."""
    D1 = w0 ** 2 - w ** 2 - 1j * s['g1'] * w
    D2 = (w0 + s['det']) ** 2 - w ** 2 - 1j * s['g2'] * w
    return D2 / (D1 * D2 - s['kappa'] ** 2)


def eit_poles(w0, s):
    """The four material poles (complex w) of one EIT layer with bright resonance w0."""
    p1 = np.array([-1, -1j * s['g1'], w0 ** 2])
    p2 = np.array([-1, -1j * s['g2'], (w0 + s['det']) ** 2])
    return np.roots(np.polysub(np.polymul(p1, p2), [s['kappa'] ** 2]))


def eit_design(chi, wc, shape=None, m_min=2.0, w0_range=(0.6, 1.4), n_grid=4001):
    """Per-layer EIT parameters with chi_EIT(w_c) = chi EXACTLY and every material pole
    decaying at least m_min times faster than the drive.

    Per layer two knobs: w0 (sets the PHASE of chi(w_c) via _eit_g) and wp2 > 0 (sets |chi|).
    All roots w0 of arg _eit_g(w_c; w0) = arg chi in w0_range*a are found (grid + brentq); the
    one with margin >= m_min and the SMALLEST wp2 is kept. chi = 0 layers are vacuum (wp2 = 0).
    Returns dict(wp2, w0, margin, ok) plus the shape entries (scalars)."""
    from scipy.optimize import brentq
    s = shape or eit_shape(wc)
    a, b = wc.real, abs(wc.imag)
    W = a * np.linspace(*w0_range, n_grid)
    G = _eit_g(wc, W, s)
    M = np.array([(-eit_poles(w, s).imag).min() / b for w in W])
    chi = np.atleast_1d(np.asarray(chi, complex))
    wp2 = np.zeros(chi.size); w0 = np.full(chi.size, a); mar = np.full(chi.size, np.inf)
    for i, x in enumerate(chi):
        if abs(x) < 1e-14:
            continue
        f = np.angle(G * np.conj(x))
        cand = np.where((np.sign(f[:-1]) != np.sign(f[1:])) & (np.abs(f[:-1]) < 1.5)
                        & (np.abs(f[1:]) < 1.5) & (np.minimum(M[:-1], M[1:]) >= m_min))[0]
        best = None
        for j in cand:
            r = brentq(lambda w: np.angle(_eit_g(wc, w, s) * np.conj(x)), W[j], W[j + 1], xtol=1e-14)
            q = x / _eit_g(wc, r, s)
            m = (-eit_poles(r, s).imag).min() / b
            if m >= m_min and (best is None or q.real < best[0]):
                best = (q.real, r, m)
        if best is None:
            raise ValueError(f'eit_design: no w0 in {w0_range} reaches chi = {x:.3g} with margin {m_min}')
        wp2[i], w0[i], mar[i] = best
    return dict(wp2=wp2, w0=w0, margin=mar, ok=True, **s)


def eps_eit(w, p):
    """eps(w) of every EIT layer: shape (len(w), n_layers); w may be complex."""
    w = np.asarray(w, dtype=complex)[:, None]
    return 1 + p['wp2'][None, :] * _eit_g(w, p['w0'][None, :], p)


def fano2_design(chi, wc, g1=10.0, g2=4.0, frac=0.9, w0_range=(0.6, 1.4), n_grid=4001):
    """Per-layer two-line parameters, chi(w) = f1/D1 - f2/D2 with D_i = w0^2 - w^2 - i g_i|b| w
    (common w0), g1 > g2 > 2 in units of |b| -> both poles decay faster than the drive
    (margin g2/2). Real-axis passivity:  Im chi = w [f1 g1/|D1|^2 - f2 g2/|D2|^2] >= 0 for all
    real w  <=>  f2/f1 <= g2/g1 (exact, from |D_i|^2 = X + g_i^2 w^2 with X >= 0).
    The hole depth is fixed, f2 = frac (g2/g1) f1, so passivity holds by construction; then, as in
    eit_design, w0 sets the phase of chi(w_c) and f1 > 0 its size (root-find, least f1 kept).
    (A free f2 solved by a 2x2 system per w0 fails as eta -> 0: pure gain is then reachable only
    at the w0 where that system is singular.)"""
    from scipy.optimize import brentq
    a, b = wc.real, abs(wc.imag)
    G1, G2 = g1 * b, g2 * b
    rho = frac * G2 / G1
    gfun = lambda w0: 1 / (w0 ** 2 - wc ** 2 - 1j * G1 * wc) - rho / (w0 ** 2 - wc ** 2 - 1j * G2 * wc)
    W = a * np.linspace(*w0_range, n_grid)
    G = gfun(W)
    chi = np.atleast_1d(np.asarray(chi, complex))
    f1 = np.zeros(chi.size); w0 = np.full(chi.size, a)
    for i, x in enumerate(chi):
        if abs(x) < 1e-14:
            continue
        f = np.angle(G * np.conj(x))
        cand = np.where((np.sign(f[:-1]) != np.sign(f[1:])) & (np.abs(f[:-1]) < 1.5)
                        & (np.abs(f[1:]) < 1.5))[0]
        best = None
        for j in cand:
            r = brentq(lambda w: np.angle(gfun(w) * np.conj(x)), W[j], W[j + 1], xtol=1e-14)
            q = (x / gfun(r)).real
            if q > 0 and (best is None or q < best[0]):
                best = (q, r)
        if best is None:
            raise ValueError(f'fano2_design: chi = {x:.3g} unreachable in {w0_range}')
        f1[i], w0[i] = best
    return dict(f1=f1, f2=rho * f1, w0=w0, g1=G1, g2=G2, margin=g2 / 2, ok=True)


def eps_fano2(w, p):
    w = np.asarray(w, dtype=complex)[:, None]
    D = lambda g: p['w0'][None, :] ** 2 - w ** 2 - 1j * g * w
    return 1 + p['f1'][None, :] / D(p['g1']) - p['f2'][None, :] / D(p['g2'])


def rt_model(eps_fn, p, d, w, chunk=4096, **kw):
    """rt() for any per-layer dispersive model eps_fn(w, p) -> (nw, N), chunked in frequency."""
    w = np.atleast_1d(np.asarray(w, dtype=complex))
    parts = [rt(eps_fn(w[i:i + chunk], p), d, w[i:i + chunk], **kw) for i in range(0, w.size, chunk)]
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
