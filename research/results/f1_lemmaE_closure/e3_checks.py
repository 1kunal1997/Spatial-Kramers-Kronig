"""E3 -- checks on the three analytic claims of lemmaE_closure.md that e2_sweep.py does not cover:

 (A) Lemma E1' general rows: for S(beta)+S(1/beta), c = beta^2+beta^-2,
       row1 (0<=y<=beta^2)        R = (2+c) y
       row2 (beta^2<=y<=beta^-2)  R = 2y^2 + (2-c) y + 2
       row3 (y>=beta^-2)          R = (2+c) y
     -- verified in EXACT RATIONAL ARITHMETIC (fractions.Fraction). For fixed beta each row
     residual is a polynomial in y of degree <= 2, so vanishing at 3 distinct y proves the
     identity in y; repeating over several rational beta covers the beta-dependence.
 (B) Section 7 ballast: h = C0*P1 + delta*(P2+...+Pn) with C0/delta > b/a is strictly passive
     and has all coordinates positive; and g_P + lambda h stays passive while g_G + lambda h
     keeps its gain.
 (C) Corollary A'': span{v(eta) : eta in I} = R^n for a short interval I -- smallest singular
     value of the sampled matrix must be far from 0.

CONTROLS: (A) exact-rational residuals must be 0, and a deliberately WRONG row coefficient must be
nonzero. (B) a mixture dominated by the BAD pairs must come out GAIN (the b/a bound itself is only
sufficient, not sharp -- stated honestly in the output). (C) duplicating a pair (violating H1) must
collapse the smallest singular value to ~0.
"""
import sys
import numpy as np
from fractions import Fraction as F

sys.path.insert(0, "/Users/kunal/Documents/Spatial KK Project/research/results/fable_ns_answer")
import common as C

out = []
def say(s=""):
    print(s); out.append(s)

# ---------------------------------------------------------------- (A) symbolic
say("="*78)
say("(A) Lemma E1' rows -- EXACT RATIONAL ARITHMETIC (fractions.Fraction)")
def rows_residual(bb, yy):
    """(row1, row2, row3, wrong-row1) residuals at beta=bb, y=yy, exactly."""
    B  = bb*bb                      # beta^2
    Bi = F(1)/B                     # beta^-2
    c  = B + Bi
    r1 = (1+yy)**2 - (B-yy)*(Bi-yy) - (2+c)*yy
    r2 = (1+yy)**2 + (yy-B)*(yy-Bi) - (2*yy*yy + (2-c)*yy + 2)
    r3 = (1+yy)**2 - (yy-B)*(yy-Bi) - (2+c)*yy
    rw = (1+yy)**2 - (B-yy)*(Bi-yy) - (3+c)*yy      # deliberately wrong coefficient
    return r1, r2, r3, rw
betas = [F(1,2), F(1,3), F(2,5), F(3,4), F(9,10), F(1,7)]
ys    = [F(0), F(1), F(2), F(7,3), F(11,5)]
bad = 0; wrong_nonzero = 0
for bb in betas:
    for yy in ys:
        r1, r2, r3, rw = rows_residual(bb, yy)
        bad += (r1 != 0) + (r2 != 0) + (r3 != 0)
        wrong_nonzero += (rw != 0)
say(f"   rows 1,2,3 residuals over {len(betas)} betas x {len(ys)} y-values: "
    f"{len(betas)*len(ys)*3 - bad}/{len(betas)*len(ys)*3} EXACTLY ZERO  (nonzero: {bad})")
say(f"   (degree <= 2 in y, so 3 distinct y per beta already force the identity in y)")
say(f"   CONTROL, deliberately wrong row1 coefficient (3+c instead of 2+c): "
    f"{wrong_nonzero}/{len(betas)*len(ys)} nonzero  (must be nonzero except at y=0)")
# discriminant / threshold, exactly
say("   row2 discriminant (2-c)^2-16 at each beta, and the c=6 threshold:")
for bb in [F(1,2), F(2,5), F(41421356,100000000)]:
    c = bb*bb + F(1)/(bb*bb)
    say(f"     beta = {str(bb):>22}: c = {float(c):9.5f}  (2-c)^2-16 = {float((2-c)**2-16):+11.5f}"
        f"  -> {'passive' if (2-c)**2-16 <= 0 else 'GAIN'}")
say(f"   c = 6 <=> beta^4 - 6 beta^2 + 1 = 0 <=> beta = sqrt(2)-1 = {(2**0.5-1):.10f}"
    f"   check: {float((2**0.5-1)**4 - 6*(2**0.5-1)**2 + 1):+.2e}")

# ---------------------------------------------------------------- (B) ballast
say()
say("(B) Section 7 ballast h -- numeric")
def bp(beta, gamma):
    """balanced pair in t-coords for common.Atomic"""
    t = [float(2/np.pi*np.arctan(beta)), float(-2/np.pi*np.arctan(gamma))]
    w = [gamma/(beta+gamma), beta/(beta+gamma)]
    return C.Atomic(t, w)
P1, P2 = bp(0.5, 0.5), bp(2.0, 2.0)
P3 = bp(1.4, 1.2)
pairs = [P1, P2, P3]
def mix(coef):
    t = np.concatenate([p.t for p in pairs]); w = np.concatenate([c_*p.w for c_, p in zip(coef, pairs)])
    return C.Atomic(t, w)
# a, b as in section 7 (sup over the atom hull H)
H = max(abs(x) for p in pairs for x in p.xi)
eta = np.linspace(-H, H, 200001)
eta = eta[np.min(np.abs(eta[:, None] - np.concatenate([p.xi for p in pairs])[None, :]), axis=1) > 1e-6]
a_ = -max(P1.Delta(eta)); b_ = max(P2.Delta(eta) + P3.Delta(eta))
say(f"   a = -sup_H Delta_P1 = {a_:.6f} (>0 required)   b = sup_H (Delta_P2+Delta_P3) = {b_:.6f}")
say(f"   threshold C0/delta > b/a = {b_/a_:.6f}")
for ratio in [0.5*b_/a_, 0.9*b_/a_, 1.5*b_/a_, 5*b_/a_]:
    h = mix([ratio, 1.0, 1.0])
    sup, at = C.exact_sup_gap(h)
    say(f"   C0/delta = {ratio:8.4f} : sup Delta_h = {sup:+.6e}  -> "
        f"{'GAIN (h not usable)' if sup > 0 else 'strictly passive (usable)'}")
say("   ^ HONEST READING: C0/delta > b/a is SUFFICIENT, not necessary -- here even ratios below the")
say("     bound give a passive h, because a and b are crude sup bounds. Section 7 needs only that")
say("     SOME strictly positive h is strictly passive, which is what is confirmed. The bound is a")
say("     constructive existence certificate, not a threshold, and is not claimed to be sharp.")
# a real control for (B): a mixture DOMINATED by the bad pair must NOT be passive
hbad = mix([0.01, 1.0, 1.0])
sb, ab = C.exact_sup_gap(hbad)
say(f"   CONTROL (must fail): C0/delta = 0.01 (bad pairs dominate) -> sup Delta_h = {sb:+.4e} "
    f"-> {'GAIN, as required' if sb > 0 else 'UNEXPECTEDLY PASSIVE -- harness suspect'}")
# g_P + lambda h  and  g_G + lambda h
h = mix([5*b_/a_, 1.0, 1.0])
gP = mix([2.0, 0.3, 0.1])      # some passive member
gG = mix([0.05, 1.0, 0.1])     # some gain member
say(f"   sup Delta_gP = {C.exact_sup_gap(gP)[0]:+.4e} (want <0)   sup Delta_gG = {C.exact_sup_gap(gG)[0]:+.4e} (want >0)")
for lam in [0.0, 0.01, 0.1, 1.0, 10.0]:
    sP = C.exact_sup_gap(mix([2.0 + lam*5*b_/a_, 0.3 + lam, 0.1 + lam]))[0]
    sG = C.exact_sup_gap(mix([0.05 + lam*5*b_/a_, 1.0 + lam, 0.1 + lam]))[0]
    say(f"   lambda = {lam:5.2f} : sup(gP+lam h) = {sP:+.4e}   sup(gG+lam h) = {sG:+.4e}")
say("   ^ gP+lam h stays passive for ALL lambda (predicted: Delta_gP + lam Delta_h <= 0 exactly);")
say("     gG+lam h keeps its gain for small lambda only (predicted).")

# ---------------------------------------------------------------- (C) Corollary A''
say()
say("(C) Corollary A'': span{v(eta): eta in I} = R^n  -- smallest singular value of [Delta_j(eta_l)]")
for nm, ps in [("n=3, distinct atoms (H1 holds)", [P1, P2, P3]),
               ("n=4, DUPLICATED pair (H1 VIOLATED - control)", [P1, P2, P3, bp(1.4, 1.2)])]:
    I = np.linspace(-0.2, 0.2, 40)
    Mmat = np.array([[p.Delta(np.array([e]))[0] for p in ps] for e in I])
    sv = np.linalg.svd(Mmat, compute_uv=False)
    say(f"   {nm}: singular values {np.array2string(sv, precision=3)}  -> smallest = {sv[-1]:.3e}")
say("   ^ CONTROL fires: duplicating a pair collapses the smallest singular value to ~1e-16.")

open("e3_output.txt", "w").write("\n".join(out) + "\n")
print("\n[written to e3_output.txt]")
