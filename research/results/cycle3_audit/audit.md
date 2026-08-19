# Cycle 3 audit — C33 (240/240), C34 (DC-offset mechanism), C35 (skew tolerance = 0)

**VERDICT: CLEAN** on all three targets. No finding changes any claim's number or verdict.
This is the third round of scrutiny on this line of work (auditor cycle-2 falsified a
diagnosis, theorist cycle-2 independently reproduced two of those falsifications and fixed
the underlying defect) — this cycle's job was to check whether the *fix* itself is sound,
using constructions independent of the theorist's own code wherever feasible, per
PROTOCOL.md's common-mode caution.

All scratch scripts are in `research/results/cycle3_audit/`. Nothing in `research/theory/`
or elsewhere was modified.

---

## TARGET 1 — C33: "(L1) corpus agreement is 240/240"

### 1a. The trigamma-DFT Clausen identity — VERIFIED, independently, exact to machine precision

Checked the identity `Cl2(2πk/M) = Σ_{r=1}^{M-1} sin(2πrk/M)·ψ₁(r/M)/M²` with a **from-scratch
matrix-vector implementation** (not `t3_clausen.py`'s FFT-aliased version) against **mpmath's
built-in Clausen function** `mp.clsin(2,θ)` (not `t3_clausen.py`'s own reference values), at
M = 12, 37, 100, 251, 1000, 4001 (mixed parities, non-powers-of-two included). Agreement is
**2e-16 to 4e-15 at every M tested** — machine precision, no drift with M.
`verify_master_identity.py`'s mpmath cross-check of Cl2(π/3) (Gieseking) and Cl2(π/2)
(Catalan) also matched to the last printed digit. **This identity is correct.**

### 1b. Per-cell exact integration (`gap_clausen`) — VERIFIED against a fully independent, high-precision, non-discretized reference

Built a **separate route with no shared code**: the analytic logistic drop density
g(t) = A·k·e^{kt}/(1+e^{kt})² integrated against the true continuous master-identity kernel
via `mpmath.quad` at 40 decimal digits (`verify_master_identity3.py`,
`verify_master_identity5.py`). This is not a rerun of DiHiTI or of `t3_clausen.py` — it never
touches a grid at all until the very last comparison step.

- First pass (nearest-grid-point comparison) showed **erratic, non-converging errors**
  (1.3e-1 → not-decaying oscillation ~1e-4 across N = 513…16385). Traced this to a **flaw in
  my own test harness**, not `gap_clausen`: comparing against the *nearest grid node* rather
  than interpolating introduces an O(h) sampling error that swamps the O(h²) quantity being
  measured. Recorded as a caution for future audits of this evaluator — a naive
  nearest-point comparison will falsely suggest `gap_clausen` is only O(h) accurate.
- With cubic-spline interpolation to the exact test points (`verify_master_identity5.py`),
  `gap_clausen` converges **cleanly at O(h²)**: error ratio **4.00, 4.00, 4.00, 4.00, 4.00,
  4.00** across seven doublings from N=513 to N=32769. This directly confirms the checkpoint's
  claimed convergence order, from an independent reference.
- Repeated the same check with DiHiTI (`tmm_h.ht_derivative`) against the *same* mpmath
  reference (`verify_dihiti_order.py`): ratio **1.96, 1.98, 1.98, 1.99, 2.00, 2.00** — clean
  O(h). This independently confirms the T3 checkpoint's threshold-attribution claim ("the
  residual ~5e-4 is DiHiTI's error, not (L1)'s") using a reference neither pipeline touches.
- Repeated the O(h²) convergence check at **non-power-of-two M** (M = N−1 = 500, 1000, 2002,
  4000 — none a power of two, guarding against a DFT-length coincidence): ratio ≈ 4.00–4.01
  throughout (`nonpow2_test.py`). The method is not secretly relying on M being FFT-friendly.

**Conclusion: the per-cell exact-Clausen integration is a genuine O(h²) scheme, not an
approximation dressed up as exact. Independently confirmed against a non-discretized
reference, at both power-of-two and non-power-of-two grid sizes.**

### 1c. Corpus reproduction — 240/240 reproduced exactly; robust across seeds, N, and grid factorization

- `python research/theory/t3_corpus.py` (unmodified): **240/240, 0.3s**, median rel err
  5.31e-4 — reproduced exactly as claimed.
- 8 other seeds × 400 profiles each (`corpus_other_seeds.py`): **3200/3200**, zero
  disagreements at any seed tested (0,1,2,3,7,11,42,999).
- 5 other N values, 200 profiles each, powers of two (`corpus_other_N.py`): **1000/1000**.
- 5 non-power-of-two N (M = 500…4998), 200 profiles each (`nonpow2_test.py`): **1000/1000**.

A fix that only works at N=2048, seed=7 would be a red flag; this one clearly does not have
that character.

### 1d. The 11 "rel=1e9" sine-family entries — confirmed genuine normalization artifact, not a masked failure

Inspected all 11 directly (`sine_artifact_check.py`). In every case DiHiTI's `e_im.max()` is
exactly 0.0 (ε″ ≤ 0 everywhere), forcing `scale = max(e_im.max(), 1e-12) = 1e-12` and any
finite absolute difference to read as ~1e9 relative. The **absolute** errors are
1.1e-3–1.8e-3, consistent with the O(h) DiHiTI baseline error at this N, and both routes agree
on the sign and rough magnitude of the minimum in every case (e.g. i=25: DiHiTI min −1.9353,
Clausen min −1.9340) — both correctly flag GAIN. **Genuine artifact, confirmed.**

### 1e. Verdict-only vs. pointwise agreement — confirmed pointwise, not just PASS/FAIL

`pointwise_check.py` computed the full-array interior RMS relative error (not just the
minimum) across all 240 profiles: **median 3.6e-4, p90 2.9e-3, max 3.7e-3** — consistent
with the claimed median 5.3e-4 pointwise stat and confirming the agreement is a genuine
pointwise match of the whole ε″(x) curve, not merely a PASS/FAIL coincidence at the minimum.

### 1f. Interior-relabel robustness (T3 addendum's claim) — reproduced: 0/240 change

Relabeling all 240 by min over |x|<0.97L instead of the whole-domain min changes **0/240**
labels (`pointwise_check.py`), matching the checkpoint's `t3_labels.py` claim exactly.

---

## TARGET 1½ — Coordinator's mid-flight concern: does the whole-domain-min convention share a blind spot with the fixed-interior-window convention flagged by red-team's C25 hardening?

Red-team's finding (near-balanced M₁→0 profiles have their true minimum migrate arbitrarily
close to the seam, and a **fixed interior window** — e.g. 0.95L — can miss it) is a real and
separate failure mode from C19's whole-domain blind spot. This needed checking against C33
specifically, since C33's corpus is exactly what's under test.

**Q1 — do the corpus label and the new evaluator's verdict use the same convention, and is
it the flawed one?**
Checked both `profile_generator.make()` (`passive=bool(e_im.min() >= -1e-3*scale)`) and
`t3_corpus.py` (`v = dom.min() >= -1e-3*scale`) directly: **both use the raw whole-domain
minimum, with no fractional/interior windowing at all.** This is *not* the fixed-interior-
window convention red-team flagged as one-sided-blind — it is the same convention C21 already
certified safe from C19's specific blind spot (endpoints are pinned to exactly 0 by
construction, so a whole-domain min cannot be fooled by the endpoint itself; it can in
principle still be fooled by *under-resolving* a dip that sits very close to, but not at, the
endpoint — which is exactly Q2).

**Q2 — do any of the 240 corpus profiles sit near the M₁≈0 knife edge where this bites?**
Direct empirical test rather than trying to compute M₁ robustly (a first attempt at computing
M₁ via a naive `Σ g(t)·tan(πt/2L)` trapezoid rediscovered the *exact* `tan(π/2)=1.6e16`
float pathology behind C34 — instructive but not usable as a diagnostic on its own). Instead,
**refined N directly** for all 240 profiles and checked whether more resolution reveals a
deeper minimum the coarser corpus grid missed (`refine_check.py`, `refine_check3.py`):

  - N=2048 (corpus) → N=16384 (8×): **0/240 label flips, 0/240 profiles with a
    >1%-of-scale deeper minimum revealed.**
  - N=2048 → N=32768 (16×): **0/240 flips, 0/240 deeper minima.**

Also tested the single most adversarially-relevant case directly: `opposite_bumps` at
α=0.2667, i.e. **exactly at the established near-critical crossover** (C18/T3-addendum). Even
at N=131072, min ε″ = −4.7e−6 (vs. exactly 0.0 at N≤32768) — a genuine but minuscule
O(h²)-scale residual, nowhere near flipping the PASS verdict, and not a "hidden deep dip."

**Conclusion: the corpus's random draws do not land near the adversarial M₁≈0 knife edge
that red-team's engineered example needed (that example required M₁ ~ 1e-4, a value reached
only by deliberately solving for a near-exact balance, not by uniform random sampling over
the stated parameter ranges). The 240/240 agreement is not an artifact of a shared blind
window.** This is a genuine, if narrow, gap in coverage: it is a statement about *this*
240-profile corpus, not a proof that no corpus profile could ever land on the knife edge. If
the corpus generator is later extended to sample near critical α values on purpose (as C18's
own crossover work does), this check should be repeated.

**Q3 — a convention sound in both directions?** Not attempted as a redesign (out of scope for
a read-only audit), but the empirical answer for *this* corpus is that the existing
whole-domain-min convention, combined with N ≥ 2048, already behaves correctly; the failure
mode is real for adversarially-tuned α (as red-team showed) but not exercised by C33's actual
test set.

**On red-team's other note** (small negative minimum at a+b=L exactly, decaying O(h²), a
quadrature artifact not a real support violation) — this is consistent with, and not in
tension with, my 1b finding that `gap_clausen` itself is a clean O(h²) scheme; a residual
O(h²) negative reading right at an exact-balance threshold is exactly what an O(h²)-accurate
evaluator applied to an exactly-zero true value should produce under floating-point
discretization. Not independently re-derived (out of scope of my three assigned targets) but
flagged as consistent.

---

## TARGET 2 — C34: the true mechanism (DC offset, tan(π/2) overflow, universal ratio, O(h) scaling, 1/3 apportionment)

Reran `t3_mechanism.py` (theorist's own script — this reproduces rather than independently
re-derives) and cross-checked the two load-bearing numeric facts independently:

- **`tan(π/2)` float pathology** — verified directly, outside any project code:
  `np.tan(np.pi/2) = 1.633123935319537e+16`, `log(...) = 37.33185619326892`. Matches the
  checkpoint's cited values exactly.
- **TEST 1** (truncation-frac sweep 0.90→0.9999): reproduced **235/233/233/233/233/233**
  agreement — flat at 233 from frac=0.95 onward, confirming truncation is not the mechanism
  (matches checkpoint and matches the auditor's own cycle-2 falsification, K18).
- **TEST 3/6** (DC-offset ratio): reproduced offset/(g(+L)+g(-L)) for the 7 offenders as
  **{0.03310, 0.03310, 0.03310, 0.03323, 0.03353, 0.03491, 0.03524}** (my run order differs
  from the checkpoint's listed order but the *set* of seven values matches the checkpoint's
  claimed {0.0331, 0.0331, 0.0331, 0.0332, 0.0335, 0.0349, 0.0352} to 4 significant figures
  in every entry.
- **TEST 5** (O(h) scaling): reproduced ratio ≈ **2.05** per halving of h, for both profiles
  tested (46, 41), across N=1024→16384 — matches the "ratios 2.05" claim exactly. The
  `offset/(h·ln(1/h))` column is *not* flat (22.8→19.7→17.2→15.2→13.6 for profile 46),
  confirming the offset is genuinely O(h) with, at most, a weak sub-leading log correction —
  consistent with the checkpoint's careful hedge ("c(h) carrying a slow log enhancement")
  rather than a literal O(h·ln 1/h) claim.
- **TEST 4** ("1/3 apportionment"): reproduced the `err_obs/pred` ratio at **−2.93 to −3.12**
  across all 7 offenders — i.e. the σ-quadrature-alone prediction accounts for almost exactly
  **1/3** of the observed offset (magnitude, opposite algebraic sign convention as coded),
  matching the checkpoint's stated apportionment precisely, not merely qualitatively.

**Is 7 samples too few to call the ratio "universal"?** It is thin, and the checkpoint is
honest that this is not claimed as a proven constant. But the ratio spans offsets from
−0.0055 to +0.289 (a 50× range) and edge sums from +0.156 to −8.74 (a 56× range) while the
*ratio itself* varies only from 0.0331 to 0.0352 — a **6% band across two-orders-of-magnitude
variation in the raw quantities**. That is a much tighter clustering than "7 samples,
coincidence" would produce; it is reasonable evidence of a genuine near-constant, though I
agree with the checkpoint's own framing that the exact constant is not load-bearing for
anything downstream (only "the offset is a DC term proportional to g(+L)+g(-L) and O(h)" is
used, and both of those are independently confirmed above).

**Conclusion: every specific numeric claim under C34 reproduces to the reported precision,
using an independent standalone check of the one non-trivial arithmetic fact (`tan(π/2)`
overflow) and re-running (not independently re-deriving) the rest. CLEAN.**

---

## TARGET 3 — C35: "skew tolerance is ZERO"

This is the highest-value target to falsify (a universal negative claim needs only one
counterexample), so I built a **fully independent construction**, not a rerun:

- **Exact point-mass (δ) atom** instead of the theorist's narrow-rectangle atom
  (`t1_verify.py` uses a 0.004-wide rect; I used the literal δ-function limit,
  `Phi_delta(θ,c)=ln|2sin((θ-c)/2)|`, `M1_delta(c)=tan(c/2)`).
- **`mpmath.clsin(2,θ)`** (the library's built-in Clausen function) instead of either of the
  project's two from-scratch Cl2 routes (trigamma-DFT or Lewin series).
- **40-digit precision**, closed-form `lam` solved algebraically from the *exact* linear
  balance equation (no discretized quadrature of `tan(φ/2)` anywhere — this is exactly the
  quantity that misbehaves near the seam under naive discretization, per Target 2, so a
  purely symbolic/closed-form route avoids that failure mode entirely rather than fighting it).
- Direct high-precision root-finding on `Φ'_μ(θ)=0` to locate the true maximum, rather than a
  coarse grid scan.

Results (`t1_independent_check.py`): **M₁ was exactly 0 to 40 digits by construction** in
every case (vs. ~1e-3 residual noise in the theorist's discretized-quadrature version of the
same check) — a materially cleaner confirmation of the "M₁=0" premise than the original.
Gains at c=-0.5π reproduced the theorist's table to 5–6 significant figures:

| q/π | my gain (δ-atom, mpmath) | theorist's gain (rect-atom, DiHiTI) |
|---|---|---|
| 0.995 | 0.09034139 | 0.090341 |
| 0.999 | 0.14203406 | 0.142034 |
| 0.9999 | 0.18630017 | 0.186300 |

Other mode positions (c=−0.25π, −0.75π, −0.1π) also matched to 5–6 significant figures. At
q=0.90π, 0.97π the gain was correctly ≈0 (θ*=−π, i.e. passive), confirming this is a genuine
PASS→FAIL transition as q sweeps toward the seam, not "always gain."

**Extended the check beyond the theorist's tested range**, per the task's instruction to hunt
for a counterexample to the *universal* claim: tested progressively smaller skew,
c = −0.30π, −0.10π, −0.03π, −0.01π (`t1_small_skew.py`, faster derivative-root-finding
variant). Found gain appearing at correspondingly larger q for every c tested down to
c=−0.01π (gain = +5.1e-5 at q=1.0π to the resolution tested), consistent with — not
contradicting — "tolerance is exactly zero but the required q→π gets more extreme as c→0."
No case was found where a skewed unimodal, exactly-balanced profile stayed strictly passive
for all q<π; every c tried eventually admits gain. **No counterexample to C35 found.**

**Conclusion: independently re-derived from a different atom construction, a different Cl2
implementation, exact (not approximately-zero) M₁ balance, and a different root-finding
method — matches the theorist's numbers to 5–6 significant figures and extends the swept
range without finding a counterexample. CLEAN.**

---

## Coverage summary

**Checked and CLEAN, this cycle:**
- C33 trigamma-DFT Cl2 identity (independent impl. + mpmath, multiple M, power-of-2 and not)
- C33 per-cell exact integration / `gap_clausen` O(h²) order (independent mpmath 40-digit
  reference, cubic-spline-interpolated comparison, power-of-2 and non-power-of-2 grids)
- C33 DiHiTI O(h) order, same independent reference
- C33 240/240 reproduction at the stated seed; robustness across 8 seeds × 400, 5×N (pow2),
  5×N (non-pow2) — 3200+1000+1000 = 5200/5200 additional agreements, zero disagreements
- C33 sine-family rel=1e9 entries — confirmed genuine normalization artifact
- C33 pointwise (not just verdict) agreement — confirmed, full-array RMS
- C33 interior-relabel robustness (0/240 change) — reproduced
- Coordinator's mid-flight concern (shared blind-window with red-team's C25 finding) —
  checked directly: same whole-domain-min convention on both sides, not the flagged
  fixed-window convention; 0/240 flips and 0/240 hidden deep minima under 8×/16× refinement;
  near-critical opposite_bumps α=0.2667 checked directly to N=131072, no hidden gain found
- C34 `tan(π/2)` float pathology — verified as a standalone fact
- C34 TEST 1 (truncation invariance), TEST 3/6 (DC-offset ratio), TEST 5 (O(h) scaling),
  TEST 4 (1/3 apportionment) — all reproduced to reported precision by rerunning the
  theorist's own script (not independently re-derived beyond the tan(π/2) fact itself)
- C35 skew-tolerance-zero claim — independently re-derived via δ-atom + mpmath Clausen +
  exact closed-form balance + derivative root-finding; matched theorist's table to 5–6
  significant figures; extended to smaller skew (down to c=−0.01π) without finding a
  counterexample

**Not checked this cycle (out of the three assigned targets, explicitly out of scope):**
- C34's TEST 2 (argmin location claim) — not independently reverified, low risk (simple
  argmin read)
- The T1 "conjectured sharpening" (sup = ln sec(c/2)) — explicitly marked unverified by the
  theorist itself; not in scope of C35's core claim, not audited here
- Any claim outside C33/C34/C35 (C1–C32 minus what cycle-2's auditor already covered)
- Red-team's C25-hardening report itself (`results/cycle3_c25_hardening/redteam.md`) — only
  the two specific findings relayed by the coordinator were checked against C33's corpus;
  the hardening report's own internal correctness was not re-audited

## Severity ranking

No finding at any severity — clean audit. The one near-miss worth flagging for future
attention (not a finding against any claim, since it changes no number): the corpus's
whole-domain-min convention is *empirically* safe for these 240 profiles but is not proven
safe in general against red-team's engineered M₁≈0 near-seam construction; if the corpus
generator is ever extended to intentionally sample near critical/balanced parameter regions,
this should be rechecked rather than assumed.
