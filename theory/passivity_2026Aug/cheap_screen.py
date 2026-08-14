"""Two questions.

(A) Can cheap, no-HT tests on eps' predict whether eps'' has gain?
    Screen = three necessary conditions, none of which needs a Hilbert transform:
      P: parity      - eps' even about the centre  => eps'' odd => mean 0 => gain
      A: area        - |int (eps' - midpoint) dx| large => 1/x tail => gain
      S: slope match - |slope_L - slope_R| large => no single Nevanlinna D => gain
    Score against ground truth min(eps'') >= 0.  Report the confusion matrix.

(B) Is the D-blindness a PHYSICAL failure mode, or only a mathematical one?
    A real coating is constant outside [-L, L], so its asymptotic linear growth
    is D = 0 identically.  Does the blindness ever fire on such a profile?
"""
import sys
sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project")
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid
import tmm_helper as tmm_h

nb, eps_bg, eps_air, kappa, L, N = 1.7, 1.7**2, 1.0, 4.0, 2.5, 4096
xx = np.linspace(-L, L, N)
A_bump = eps_bg - eps_air


def dihiti(x_, e):
    n = len(e)
    v = np.imag(hilbert(np.gradient(e, x_)))
    c = cumulative_trapezoid(v, x_, initial=0)
    return c - np.linspace(c[0], c[-1], n)


def gauss(x_, x0, s):
    return np.exp(-((x_ - x0) / s)**2)


# ----------------------------------------------------------------- profiles
def make_tanh(mL, mR):
    b, d = (mL + mR) / 2, (mR - mL) / 2
    a = (eps_bg + mL * L + eps_air - mR * L) / 2
    c = (eps_air - mR * L - eps_bg - mL * L) / 2
    return a + b * xx + c * np.tanh(kappa * xx) + d * xx * np.tanh(kappa * xx)


def make_tri(s):
    return eps_air + s * (np.sqrt(L**2 + .15**2) - np.sqrt(xx**2 + .15**2))


def make_cos(beta):
    return eps_air + (A_bump / 2) * (1 + np.cos(np.pi * xx / L)) * (1 + beta * xx)


e_log = tmm_h.logistic(xx, 4.0, nb)
bump_c = A_bump * gauss(xx, 0.0, 0.6)
bL = A_bump * gauss(xx, -0.9, 0.45)
bR = A_bump * gauss(xx, +0.9, 0.45)


def inv_ht(eim):
    return eps_air - np.imag(hilbert(eim))


C = []
C.append(('logistic k=4', e_log))
C.append(('Lorentzian A=1 g=0.5', np.real(tmm_h.eps(xx, 1.0, 0.5, nb))))
C.append(('Lorentzian A=5 g=0.3', np.real(tmm_h.eps(xx, 5.0, 0.3, nb))))
C.append(('bwd smooth mesa', inv_ht(1.2 * (np.tanh(4 * (xx + 1.2)) - np.tanh(4 * (xx - 1.2))) / 2)))
C.append(('bwd symmetric bump', inv_ht(1.5 * gauss(xx, 0., .8))))
C.append(('bwd double peak', inv_ht(.8 * gauss(xx, -.9, .5) + 1.4 * gauss(xx, .5, .5))))
for s in (0.5, 2.0, 3.5):
    C.append((f'triangle s={s}', make_tri(s)))
for b in (0.0, 0.2, 0.38):
    C.append((f'cosine b={b}', make_cos(b)))
for mL, mR in ((2., 1.), (4., -3.), (6., -2.5)):
    C.append((f'tanh mL={mL} mR={mR}', make_tanh(mL, mR)))
C.append(('odd descending (high L)', eps_bg - .945 * np.sin(np.pi * xx / L) - .945))
C.append(('odd ascending  (low L)', eps_bg + .945 * np.sin(np.pi * xx / L) - .945))
for al in (0.1, 0.3, 0.5, 1.0, 2.0):
    C.append((f'logistic+bump a={al}', e_log + al * bump_c))
for al in (0.0, 0.3, 0.6, 0.7, 1.0, 1.5):
    C.append((f'opp bumps a={al}', e_log + al * (bL - bR)))


# ------------------------------------------------------------ cheap screen
def screen(e):
    """Three no-HT scores, each normalised to be dimensionless."""
    span = e.max() - e.min()
    if span < 1e-12:
        span = 1.0
    mid = (e[0] + e[-1]) / 2
    w = max(8, N // 12)
    slL = np.polyfit(xx[:w], e[:w], 1)[0]
    slR = np.polyfit(xx[-w:], e[-w:], 1)[0]
    return dict(
        P=np.max(np.abs(e - e[::-1])) / span,          # ~0  => even  => gain
        A=abs(np.trapezoid(e - mid, xx)) / (span * L),  # large => gain
        S=abs(slL - slR) / (span / L),                  # large => gain
    )


TH_P, TH_A, TH_S = 0.05, 0.05, 0.30

rows = []
for lb, e in C:
    s = screen(e)
    eim = dihiti(xx, e)
    truth_pass = eim.min() >= -1e-3
    flags = []
    if s['P'] < TH_P:
        flags.append('P')
    if s['A'] > TH_A:
        flags.append('A')
    if s['S'] > TH_S:
        flags.append('S')
    rows.append((lb, s, eim.min(), truth_pass, flags))

print('=' * 96)
print(f"{'profile':<26}{'P(parity)':>10}{'A(area)':>9}{'S(slope)':>10}"
      f"{'flags':>8}{'screen':>9}{'min eps''':>12}{'truth':>8}")
print('=' * 96)
for lb, s, mn, tp, fl in rows:
    scr_pass = len(fl) == 0
    mark = '' if scr_pass == tp else '   <-- MISS'
    print(f"{lb:<26}{s['P']:>10.3f}{s['A']:>9.3f}{s['S']:>10.3f}"
          f"{','.join(fl) if fl else '-':>8}"
          f"{'PASS' if scr_pass else 'GAIN':>9}{mn:>12.4f}"
          f"{'PASS' if tp else 'GAIN':>8}{mark}")
print('=' * 96)

tp_ = sum(1 for _, _, _, t, f in rows if not t and f)      # gain, caught
fn_ = sum(1 for _, _, _, t, f in rows if not t and not f)  # gain, missed
tn_ = sum(1 for _, _, _, t, f in rows if t and not f)      # passive, clean
fp_ = sum(1 for _, _, _, t, f in rows if t and f)          # passive, false alarm
print(f"\nGain caught by screen : {tp_}/{tp_+fn_}"
      f"   MISSED: {fn_}")
print(f"Passive kept clean    : {tn_}/{tn_+fp_}   false alarms: {fp_}")

# ------------------------------------------- (B) is D-blindness physical?
print('\n' + '=' * 96)
print("(B) D-blindness: mathematical curiosity or physical failure mode?")
print('=' * 96)
print("A physical coating is CONSTANT outside [-L,L], so its asymptotic linear")
print("growth is D=0 by construction. The Nevanlinna D-term can only be nonzero")
print("for a profile that ramps linearly to x = +/- infinity.\n")

ramp = e_log + 1.0 * xx
print(f"  logistic + 1.0*x, taken literally on all of R : D = +1.0  -> Nevanlinna OK")
print(f"  logistic - 1.0*x, taken literally on all of R : D = -1.0  -> NOT Nevanlinna")
print(f"  either one, CLAMPED outside [-L,L] (physical) : D =  0    -> D-test is vacuous")
print()
print("  eps'' from DiHiTI is identical in all three cases:")
for lbl, prof in (('D=+1', e_log + xx), ('D=-1', e_log - xx), ('D= 0', e_log)):
    print(f"    {lbl}:  min(eps'')={dihiti(xx, prof).min():+.6f}   "
          f"max|diff vs D=0| = {np.abs(dihiti(xx,prof)-dihiti(xx,e_log)).max():.2e}")
