# Cycle 2 Audit — checkpoint / working state

**GOAL.** Independently audit three cycle-1 result clusters before they climb the claim ladder:
(1) C28 — theorist's self-diagnosis of (L1)'s 233/240 corpus defect as a ξ-space quadrature/
truncation artifact; (2) C18/C19 — numericist's α_c crossover and blind-indicator claims, plus C13
endpoint exactness; (3) C21/C22 — red-team's rank-deficiency and DiHiTI-reduction claims, and
whether K13's unsound-Sobolev-certificate reasoning leaked elsewhere in its report.
Read-only. Scratch scripts live in `research/results/cycle2_audit/`; no existing file touched.

---

## ESTABLISHED SO FAR

### 1. C28 diagnosis — **VERIFIED WRONG. This is the headline finding.**

Independently reimplemented the domination gap Δ(η) = ∫ln|η−ξ(t)|g(t)dt − (D/2)ln(1+η²) from the
paper definition (`research/results/cycle2_audit/audit_c28b.py`), using the same singular-cell
regularization technique as `h9_fixes.py` (needed — a naive kernel without it gives spurious −inf
diagonal terms and reproduces only 153/240, not 233/240; recorded as a ruled-out red herring below).
With this kernel I **reproduce the reported 233/240 exactly** at the 0.97L cutoff (verified-numerically),
confirming the corpus-agreement number itself is not fabricated.

**Then I swept the truncation fraction from 0.90 to 0.99999 (a ~30,000× tightening of the excluded
seam region) on the 7 disagreeing profiles.** The diagnosis in `h9_potential_domination.md` line
477-485 and `h9_fixes.py` predicts the disagreement should shrink/vanish as the cutoff approaches L,
because "the near-seam region where the minimum lives" gets included. **It does not:**

| profile | dom_min @ 0.97 | dom_min @ 0.99999 | change |
|---|---|---|---|
| tanh_slopes | +0.06035 | +0.06035 | **exactly 0** |
| opposite_bumps | +0.00857 | +0.00857 | **exactly 0** |
| tanh_slopes | +0.11374 | +0.11374 | **exactly 0** |
| tanh_slopes | +0.10133 | +0.10133 | **exactly 0** |
| lorentzian | +0.00411 | +0.00411 | **exactly 0** |
| logistic | −0.00485 | −0.00620 | moves, but *away* from agreement, plateaus nonzero |
| logistic | −0.00394 | −0.00529 | moves, but *away* from agreement, plateaus nonzero |

Full-corpus agreement count vs. frac: **235/240 → 233/240 → 233/240 → ... → 233/240** (frac =
0.90, 0.95, 0.97, ..., 0.99999) — flat from 0.95 onward. If truncation of the near-seam region were
the cause, agreement should trend toward 240/240 as frac → 1. It does not move at all past frac=0.95.

**Second falsification, on the diagnosis's own stated mechanism ("edge-slope-unmatched families
g(+L) ≠ g(−L)"):** computed g(±L) directly for all 7 disagreeing profiles. Only the 3 `tanh_slopes`
instances actually have g(+L) ≠ g(−L) (mismatch 1.16–1.44). The other 4 — both `logistic` cases and
the `opposite_bumps` and `lorentzian` cases — have **g(+L) = g(−L) to machine precision** (mismatch
0.00000). The stated premise ("all 7 disagreements are edge-slope-unmatched families") is **factually
false for 4 of the 7**.

**Third check — location of the Δ(η) maximizer** (the thing 0.97L truncation would have to be
excluding for the diagnosis to hold): for `opposite_bumps`, `lorentzian`, and one `tanh_slopes` case,
the maximizer sits at x/L ≈ 0.80–0.91 — **well inside** the 0.97L window already, not excluded by it
at all. Only the two `logistic` cases have a maximizer near the seam (x/L ≈ 0.999), and even for
those, tightening the window to 0.99999L leaves a stable **nonzero** residual disagreement rather than
closing it.

**Verdict: the diagnosis in C28 is incorrect.** The 233/240 defect is not a truncation/quadrature
artifact of the 0.97L cutoff. It survives a 30,000× tightening unchanged for 5/7 profiles, and the
stated causal premise (universal edge-slope mismatch) is empirically false for 4/7 of the named
disagreers. **Consequence: (L1)'s corpus agreement should not be reported as "233/240, cause
identified, 20-minute fix pending" — the cause is not identified.** Whatever is actually causing 5 of
these 7 (all with a genuine, non-negligible DiHiTI min ε″ — e.g. tanh_slopes at −0.229, −0.168,
−0.177, well outside noise) to be misclassified as passive by the domination-gap test is still open.
This is either a real defect in (L1) on these families, or a bug elsewhere in the corpus-agreement
harness (e.g. how `dl.max()` searches η — restricted to grid-derived η only, not exhaustively over
ℝ). I did not have time to pin down the *true* cause before this checkpoint; see NEXT STEP.

Ruled out as the explanation: naive (unregularized) kernel quadrature — that's a different, unrelated
defect (gives 153/240) I hit first and discarded once I matched the theorist's actual method.

Files: `research/results/cycle2_audit/audit_c28.py` (naive kernel, superseded/kept for the record),
`research/results/cycle2_audit/audit_c28b.py` (regularized kernel, the one whose numbers are quoted
above — reruns cleanly, ~15s).

### 2. C18 / C19 / C13 (numericist) — independently verified, all CLEAN.

**C18 (α_c = 0.66623) — CLEAN.** Reproduced the bisection independently (own script, not a
re-run of theirs) with the reported N-doubling sequence; differences between successive N halve
cleanly (O(1/N) convergence: −5.17e−5, −2.58e−5, −1.29e−5, −6.4e−6 for N = 10001→160001), consistent
with genuine grid convergence, not a coincidence or a bug. **Interior-window sensitivity check (my
own addition, not in their notes): swept `edge_excl` from 0.05 to 1.15 (window half-width from
|x|≤2.45 down to |x|≤1.35, an 8× range) at N=80001 — α_c is identical to 6 decimals
(0.666236) across the entire range.** The claim that the result doesn't depend on the interior-window
definition is correct and is in fact *more* robust than their notes claimed (they didn't report this
sweep). No finding.

**C19 (blind whole-domain-min indicator) — CLEAN.** Independently rebuilt the opposite-bumps family
and confirmed: for every α in the passive regime (0.0, 0.1, 0.3, 0.5, 0.66), `min(ε″)` over the whole
domain is exactly `0.0` (float64-exact) and is attained exactly at the boundary index (x = −L, idx=0).
Matches C19 exactly.

**C13 (ε″(±L)=0) — CLEAN, and the "exact by construction vs. discretization error" distinction is
correctly drawn.** Independently verified the more interesting (pre-de-tilt, raw) endpoint claim on
an off-centre tanh step (own profile, not theirs): raw endpoint mismatch is 4.81e−4, 2.41e−4,
1.20e−4, 6.02e−5, 3.01e−5, 1.51e−5 for N = 2001→64001 — the ratio between successive halvings is
1.999, 1.999, 2.000, 2.000, 2.000: clean O(1/N) decay to zero, exactly as claimed. The post-de-tilt
value being exactly 0.0 is indeed trivial-by-construction (subtracting the linear interpolant between
the two endpoints forces this) — correctly distinguished from the nontrivial claim (raw mismatch → 0
in the continuum limit), which is the one that actually needed checking and does check out.

### 3. C21 / C22 (red-team) — independently verified, both CLEAN. K13 leak check: none found.

**C21 (rank N−2, rows 0/N−1 zero, linearity, positive interior margin) — CLEAN.** Independently
rebuilt the DiHiTI operator matrix from scratch (own script, different N ∈ {128,256,400}, different
L=1.7 from their L=1.0, standard basis vectors) and confirmed: rank = n−2 exactly at every n tested;
row 0 and row n−1 are max-abs exactly 0.0 (not just small); row 1 is not (max abs ≈0.48, so it's not
that everything is trivially zero). Also independently confirmed exact linearity of the DiHiTI map
(own random u1/u2, own coefficients a=−1.3, b=4.7): max deviation 3.6e−15 against scale 4.5, i.e.
float64-exact. Both pillars of C21 hold independently.

**C22 (DiHiTI ≡ one Hilbert transform of ramp-periodized profile, ~1.7×) — CLEAN.** Independently
reimplemented both `dihiti` and `onehilb` (own profiles: three (k,L,N) logistic combos not used by
red-team) and confirmed algebraic near-equivalence: rel-rms diff 6.1e−4 to 3.2e−3, same order as
red-team's reported 1.6e−4–3.7e−4 (my grids/profiles differ, so exact match isn't expected — but same
ballpark, confirming the effect is real and not cherry-picked). Independently timed both on my own
machine (own methodology: 5-call warmup, 50 repeats, median): dihiti 567 μs vs onehilb 358 μs → 1.58×
speedup — the same order as their reported 1.70× (numbers won't match exactly across machines/methodology,
but both land well below the mission's 3× bar for V1, which is exactly what the report concludes).
Benchmark methodology (`bench.md`) is sound: named machine/software versions, `timeit.repeat`
best-of-20 and median-of-20 reported together (not just best-of, which would be optimistic), multiple
profile families, interior-only accuracy check separate from timing. No red flags.

**K13 leak check — NONE FOUND.** Grepped `redteam.md` for every mention of `cert.py`/the Sobolev
tail bound (lines 127–130, 233, 318, 332–333, 479, 533). Every occurrence explicitly and consistently
flags the certificate as unsound/failed ("UNSOUND", "not sound as implemented", "Honest negative
result"); none of the report's other findings (C21, C22, the H1/H2/H3 verdicts, the V2 restatement)
cite or depend on `cert.py`'s numbers. The proposed *replacement* bound in the E3 spec (§4,
`sum_{|m|>n/2}|p̂_m| ≤ ‖p‖_TV/(πm)`, a total-variation tail bound) is a different, not-yet-built
construction explicitly framed as future work, not a reuse of the unsound central-difference argument.
No leak.

---

## RULED OUT

- Naive (no singular-cell regularization) reimplementation of the domination-gap kernel as the audit
  tool: gives 153/240 agreement, not 233/240, because ln|η−ξ| blows up on the diagonal (η exactly
  equals a grid ξ value when η is drawn from the same grid). Not a finding about (L1) — just the wrong
  tool. Switched to the regularized (cell-averaged) kernel matching `h9_fixes.py`'s technique, which
  reproduced 233/240 and is the one used for the real test.

## STATUS: AUDIT COMPLETE — all three priorities checked.

## NEXT STEP (for whoever picks this up)

Not required by the auditor charter, but flagged as a live, cheap lead for the theorist/next cycle:
pin down the *actual* cause of C28's 5 persistent disagreements (tanh_slopes ×3, opposite_bumps,
lorentzian). Candidate alternative hypothesis not yet tested: `dl.max()` in the corpus-agreement code
searches η only over grid-derived points (`eta_eval = tan(pi*xs[mm]/(2*Ls))`) rather than a finer or
continuous search — an under-resolved η-search would also produce a frac-invariant wrong answer if the
true maximizer of the domination gap falls between sampled η values, independent of the truncation
window. Distinguishing "law is genuinely wrong on these families" from "η-search is under-resolved" is
the next diagnostic step and is cheap (~30 min): refine the η grid independently of the t-truncation
and see if the sign flips.

## OPEN QUESTIONS / BLOCKERS

None blocking. Audit is done; report below.

---

# FINAL REPORT

**VERDICT: FINDINGS.** One significant finding (C28); everything else audited is CLEAN.

## Findings, ranked by severity

### FINDING 1 (HIGH — changes a claim's status) — C28's diagnosis is wrong

**File:** `research/theory/h9_potential_domination.md:477-485`, `research/theory/h9_fixes.py`
(block C, lines 47-65).

**What's wrong:** The theorist attributes (L1)'s 7/240 corpus disagreements to "a ξ-space quadrature
defect... my 0.97L truncation cuts off exactly the near-seam region where the minimum lives," calling
it a "20-minute fix." Independent reimplementation of the domination-gap test (matching their
singular-cell-regularized kernel, reproducing 233/240 exactly) and then sweeping the truncation
fraction from 0.90 to 0.99999 — a ~30,000× tightening of the excluded seam region — shows the
disagreement is **frac-invariant to the last decimal place** for 5 of the 7 profiles (tanh_slopes ×3,
opposite_bumps, lorentzian): the reported `dom_min` value literally does not change at all as the
truncation window is relaxed. The remaining 2 (`logistic`) do move with frac, but converge to a stable
*nonzero* residual disagreement rather than resolving. Independently locating the maximizer of the
gap function confirms it: for `opposite_bumps`, `lorentzian`, and one `tanh_slopes` instance, the
maximizer sits at x/L ≈ 0.80–0.91 — already well inside the 0.97L window, nowhere near being excluded.

**Second, independent falsification of the stated causal mechanism:** the diagnosis claims "all 7
disagreements are edge-slope-unmatched families" (g(+L) ≠ g(−L)). Directly computed g(±L) for all 7:
only the 3 `tanh_slopes` cases actually have a slope mismatch (1.16–1.44). The other 4 — both
`logistic` cases, `opposite_bumps`, and `lorentzian` — have g(+L) = g(−L) to machine precision
(mismatch 0.00000). The premise is factually false for 4/7 of the named cases.

**Concrete consequence:** C28 should not be reported (even at `conjecture`) as "cause identified,
20-minute fix pending" — that specific causal story is refuted by both a convergence test and a direct
check of its own stated mechanism. The 233/240 number itself is confirmed correct (agreement count
reproduces exactly), so C23's headline "(L1) proved, verified to 1.2e-4" is not directly touched by
this finding — but the mission's own next-step plan (`h9_potential_domination.md` §12, next-step #5)
of a "20-minute fix" via better ξ-space quadrature will not close the gap, because the disagreement
isn't a quadrature artifact. Whatever is really wrong (candidates: a genuine defect of (L1) on these
specific families, or an under-resolved η-search in the corpus harness — see NEXT STEP above) is still
open and should not be treated as identified.

**Scripts:** `research/results/cycle2_audit/audit_c28.py` (naive-kernel version, superseded — kept for
the record, reproduces only 153/240 because it lacks the singular-cell regularization, a different and
unrelated issue), `research/results/cycle2_audit/audit_c28b.py` (the regularized-kernel version whose
numbers are quoted above; reproduces 233/240 exactly, then runs the truncation sweep and the g(±L)
mismatch check). Both rerun cleanly in <20s.

## What I checked and found CLEAN (with independent verification, not just re-reading)

- **C18 (α_c = 0.66623 crossover)** — reproduced the bisection convergence sequence independently
  (own script); confirmed clean O(1/N) convergence in the successive-N differences. Additionally swept
  the interior-window half-width (`edge_excl`) from 0.05 to 1.15 — an 8× range not tested in the
  original notes — and found α_c invariant to 6 decimals across the whole range. More robust than
  claimed, not less.
- **C19 (blind whole-domain-min indicator)** — independently rebuilt the opposite-bumps family and
  confirmed min ε″ is exactly 0.0 (float64-exact) at the boundary index for every passive α tested.
- **C13 (ε″(±L)=0)** — independently verified the nontrivial half of the claim (raw, pre-de-tilt
  endpoint mismatch → 0 as N→∞) on an off-centre tanh step of my own construction: mismatch halves
  exactly (ratio 1.999–2.000) across N = 2001→64001, consistent with clean O(1/N) quadrature decay to
  zero, not a coincidence or a masked bug.
- **C21 (rank N−2, structurally-zero endpoint rows, DiHiTI linearity)** — independently rebuilt the
  operator matrix at different N and a different L than red-team used; rank = n−2 exactly, rows 0/n−1
  exactly zero (not merely small), row 1 is not zero (rules out a degenerate all-zero matrix).
  Linearity confirmed to float64 precision (3.6e−15) on independently chosen inputs.
- **C22 (DiHiTI ≡ one Hilbert transform of the ramp-periodized profile, ~1.7×)** — independently
  reimplemented both pipelines on three profiles not in red-team's test set; rel-rms agreement in the
  same 1e−4–1e−3 ballpark as claimed. Independently timed on this machine: 1.58× speedup, same order
  as the claimed 1.70×, both safely below the mission's 3× bar — consistent with the report's own
  "below V1's bar" conclusion, not overstated.
- **K13 leak check** — grepped every mention of the unsound Sobolev tail certificate (`cert.py`) across
  `redteam.md`; confirmed it is consistently flagged as unsound everywhere it's mentioned and that no
  other finding (C21, C22, the H1/H2/H3 verdicts, the V2 restatement) depends on its numbers. The E3
  future-work spec proposes a genuinely different (TV-norm) bound, not a reuse of the broken one.

## Coverage note

I did not independently re-derive or attack C1–C12, C14–C17, C20, C23–C27 — those were outside this
cycle's audit scope per the dispatch brief (three named priorities). C20 (the g-shape claim underlying
H6) was read but not separately verified beyond what C18's independent check already covers.
