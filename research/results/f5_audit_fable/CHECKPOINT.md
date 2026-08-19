# CHECKPOINT — F5 audit of fable_ns_answer (auditor-F5)

## GOAL
Numerical-hygiene audit of `research/results/fable_ns_answer/` (common.py evaluators + e0-e4
experiment scripts). Verdict per claim C48-C54, feeding C49's replicated->hardened promotion
(needs auditor CLEAN + red-team SURVIVED). Read-only on audited code; my own scratch scripts live
in the scratchpad dir (not committed) and are reproduced inline below so a successor can rerun them.

## FINDINGS SO FAR (provisional, all with independent controls run)

- **C48 (Theorem 1, density)**: CLEAN. Reran `e1_density.py` verbatim — reproduces HANDOFF numbers
  exactly. Lead #5 (non-monotone width in eta-space for eta0=8: 10.3->43.5->7.8->3.4->1.86) CONFIRMED
  as a real artifact but CONFIRMED HARMLESS: (a) converted to t(x)-space independently, width is
  monotone (0.290->0.084->0.054->0.031->0.018), matching the auditor brief's claim exactly; (b) pushed
  delta down to 1e-6, eta-space width resumes monotone shrinkage after the coarse range (1.86 -> 0.057
  at delta=1e-6), so it does eventually -> 0, just non-monotonically at coarse delta. (c) Read the
  actual Theorem 1 proof in ns_answer.md Sec 3: it only needs pointwise convergence Delta_ps(eta)->
  Delta_p0(eta) as s->0 for fixed eta != eta0, NOT monotonic shrinkage - so even the coarse-delta
  wobble would not break the proof. Bracket-expansion loop `while f(eL)>0: eL-=step; step*=1.5` in
  e1_density.py: did NOT find evidence it walks past a second sign change in any tested case (Lemma B,
  C50, proves Delta is strictly unimodal between atoms + strictly negative outside the hull for these
  balanced-pair constructions, so single sign change is guaranteed for this family). CLEAN.

- **C49 (Theorem 2, no finite battery) — witnesses k=8 and k=16**: CLEAN on the exact-evaluator side,
  with one real but non-fatal caveat on the production corroboration (see below).
  - Lead #1 (magic-number sensitivity of critical_etas): tested trim_tol in [1e-14..1e-4], imag_tol in
    [1e-11..1e-3], atom_tol in [1e-14..1e-6] on the k=16 witness (36 atoms, the most delicate case).
    Verdict (GAIN/PASSIVE) and sup Delta value are IDENTICAL to displayed precision across trim_tol
    1e-14 to 1e-8 (7 orders of magnitude); only breaks at trim_tol=1e-6 (loses digits) and 1e-4
    (wrong verdict entirely) - both are ~1e6-1e8x looser than the default 1e-12, way outside any
    plausible float64 noise floor. imag_tol and atom_tol: NO sensitivity at all across the tested
    range. Default 1e-12/1e-9/1e-12 sit deep in the safe plateau. CLEAN - not fragile to the chosen
    constants.
  - Lead #1 (does polyfromroots over 36 clustered roots lose accuracy): tested via a FULLY INDEPENDENT
    method (mpmath 50-decimal-digit precision, closed-form Delta'(eta) sign-change scan + bisection —
    NOT polynomial companion-matrix rooting at all). Result for k=16 witness: gG sup Delta =
    5.04021517805e-5 @ eta=0.956590018764909 (common.py: +5.04021518e-05 @ 0.956590 — agrees to 9
    sig figs); gP sup Delta = -5.03939353645e-5 @ eta=0.9565572488 (common.py: -5.03939354e-05 @
    0.956557 — agrees to 9 sig figs). Also verified k=8 witness the same way: matches
    +2.436530877e-2 / -2.421095177e-2 to common.py's printed 2.436531e-02/-2.421095e-02. NO accuracy
    loss found. This directly answers "is k=16 real or float noise" — it is REAL, confirmed by an
    algorithmically unrelated high-precision method. CLEAN.
  - Lead #2 (k=8 battery rank 7 not 8): confirmed the rank deficiency is in the mass-vector kernel
    construction (5 bad + 5 good pairs span only a rank-7 image under the 8 functionals), NOT in the
    functionals being read out. Independently reimplemented the 8 functional readings from the raw
    witness_k8.npz atoms/weights with fresh hand-written code (not reusing apply_battery) —
    reproduced identical readings for gG/gP to max rel diff 6.1e-16, matching the stored npz values
    exactly. So "8 functionals matched" is literally true regardless of the rank-7 kernel-construction
    detail; dim ker=3 >= 2 required is also plenty. CLEAN, rank-7 is harmless exactly as brief
    suspected.
  - Lead #6 (dead code): confirmed common.py line ~90 `Q = npoly.polymul([1.0,0.0,1.0], np.zeros(1))`
    is immediately overwritten at line 95-96. Cosmetic only, zero numeric consequence (verified by
    diffing behavior with/without that line — trivial, it's simply dead). No other dead-code
    instances found nearby in critical_etas/exact_sup_gap/exact_min_eps2.

- **C53 (O(h) production offset, "cancels in differences") — MOSTLY CONFIRMED, mechanism now PARTLY
  IDENTIFIED, with a real but bounded caveat that changes HOW the claim should be stated, not whether
  differences cancel.**
  - Independently confirmed (own script, not reusing common.py's production_min_eps2, direct calls to
    tmm_h.ht_derivative): for the k=16 witness at s=6e-4, comparing production ht_derivative output to
    the exact closed-form eps''(x) at the true argmin location x0=0.4859 (verified argmin location
    matches between exact and production at N=2^20+, resolving my initial worry it might be a
    different local extremum): the (production - exact) discrepancy at fixed x0 DOES shrink roughly
    O(h) with the DEFAULT grid="linspace" convention (np.linspace(-L,L,N)) from N=2^19..2^23: log-log
    slope 1.18, consecutive-doubling ratios 2.06/2.12/2.27/2.73 (not exactly 2 but same order,
    consistent with the HANDOFF's "halves per N-doubling" claim).
  - CRITICAL NEW FINDING (not in the original brief, found via lead-#1's "vary the construction"
    mandate applied here instead): switching to the OTHER grid option already present in
    common.py's production_min_eps2 (grid="periodic", i.e. x = -L + 2L*arange(N)/N, no duplicated
    endpoint) gives QUALITATIVELY DIFFERENT behavior: the offset does NOT vanish as N->infinity. It
    decays fast from N=2048 (-3.47e-5) down to N~131072 (-8.4e-7) then PLATEAUS at a fixed floor
    ~-8.3e-7 all the way to N=2^22 (checked -8.317e-7 at 2^22). This floor is grid-convention-
    independent in the limit (I extended the linspace sweep to N=2^25 and it crosses zero around
    2^24-2^25 then heads toward the same ~-8e-7 magnitude/sign region), consistent with TWO additive
    error sources: (1) an O(h) term specific to the linspace-duplicated-endpoint convention (this is
    literally the PROTOCOL-flagged historical bug pattern: np.linspace(-L,L,N) creates a period
    mismatch of exactly dx under FFT periodic continuation - I verified this directly: N*dx-2 = +dx
    for linspace vs exactly 0.000e+00 for periodic, at N=2048, 2049, 4096); this term dominates at
    corpus-historical N and is what "halves per doubling"; (2) a smaller, N-independent residual
    floor (~-8e-7 for this witness/s combo) shared by both conventions - presumably a genuine
    algorithm-level bias in ht_derivative's erf-smoothed-staircase + cumulative_trapezoid + linear
    detrend pipeline, not yet identified (matches HANDOFF's "mechanism unidentified" honesty).
  - "Cancels in differences" claim: RECONFIRMED independently and robustly with BOTH grid
    conventions - production(gG)-production(gP) matches the exact closed-form difference to 4 sig
    figs (ratio 1.0002) at every N from 2^19-2^22, for both linspace and periodic. This is the part
    that actually matters for C49's production corroboration and it is genuinely solid.
  - Practical magnitude finding worth flagging: at N=2048 (explicitly named in HANDOFF as
    "corpus-historical"), linspace gives an ABSOLUTE offset of +8.03e-3 (checked over a 5-point
    window around the argmin to rule out a single-point fluke - consistent +8.02e-3 to +8.05e-3
    across the window), while periodic gives -3.47e-5 at the SAME N - i.e. the periodic convention is
    ~230x more accurate at N=2048 for this witness. HANDOFF's stated "~1e-3 scale at N=2048-2^14" is
    roughly right in magnitude for what they likely measured (their own C0.3/e2 controls all use
    linspace or a fixed 60000-periodic in different combos) but understates the true linspace-N=2048
    error I measured (8e-3, not 1e-3) for the ACTUAL k=16 witness/s=6e-4 combo - worth a note but not
    a verdict-changer since C49's actual witness verdicts (C49) never rely on N=2048; they explicitly
    use N=2^20-2^22 where linspace offset (~1e-5) is well below the true signal (~1.6e-5), which is
    still a fairly thin margin (ratio offset/signal ~ 0.93 at N=2^20) - see IN PROGRESS below, this
    needs one more check: does the GAIN verdict for gG at N=2^20 survive on the linspace grid only
    because of favorable sign, i.e. is it fragile to N choice below 2^20? (I have the data: at N=2^19
    the offset +3.06e-5 EXCEEDS the true signal -1.6e-5 in magnitude, which would flip the sign to
    false-PASSIVE - i.e. N=2^19 linspace would have given the WRONG verdict for gG. The authors did
    not test N=2^19, only 2^20 and 2^22, and got lucky/careful in stopping at N=2^20.) This is the
    single most important thing to write up carefully in the final report - it's a real fragility in
    the "production-confirmed" claim for the k=16 witness (not in the underlying math, which is
    exact-evaluator-based and unaffected).

## CLEARED (checked and harmless, no need to recheck)
- Lead #1 magic numbers (trim/imag/atom tolerances) in critical_etas — robust over 6-7 orders of
  magnitude.
- Lead #1 polyfromroots accuracy for 36 clustered atoms — verified via independent mpmath route,
  matches to 9-10 sig figs.
- Lead #2 k=8 rank-7 — harmless, 8 functionals genuinely match independently reproduced.
- Lead #5 e1_density non-monotone eta-space width — confirmed artifact, confirmed harmless (matches
  brief's own diagnosis), theorem proof doesn't need monotonicity.
- Lead #6 dead code at common.py ~line 90 — cosmetic, zero consequence, no siblings found nearby.
- e0_controls.py fully rerun, reproduces all HANDOFF numbers exactly (Cor 4.A threshold 1.000000000
  x3, C40 slopes to 4 digits, lambda* 0.2620/0.2778 matching red-team/handA).
- e3_n1_closure.py lam_star grid-based minimization (400001-pt t-space-uniform grid, NOT eta-space -
  good design choice, avoids the Cayley-stretch issue): independently refined the most sensitive
  case (corner (-0.4,0.9)+bulk(-0.24,0.74), lambda*=0.0658) with a 10x-denser grid AND a local
  scipy.optimize.minimize_scalar refinement - both agree to 5 sig figs (0.065757). Grid resolution is
  adequate, not under-resolving the sharp near-threshold falloff. CLEAN.
- e4_lemmaE.py "contiguous window [1.45,2.96], no gaps" claim: verified directly (not just trusting
  the diff-based gap check in the script) that literally EVERY grid theta in [1.45,2.96] (step 0.01,
  152 points) was accepted and ZERO outside that range were accepted - the contiguous-window claim is
  robust at this resolution, not a gap-check-logic artifact. Correctly labeled "numerically certified,
  not proved" in ns_answer.md - not overclaimed.

## NOT YET DONE
- C50, C51, C52, C54 not yet independently audited (C51's e3 lam_star spot-checked above as a
  byproduct of the N1 lead but not systematically; C50/C52/C54 are proof-only claims with no
  dedicated numerical script - need to at least confirm there's no numerical claim hiding in them
  that needs a control).
- Need to write up the N=2^19-vs-2^20 fragility finding above formally with one more control: rerun
  the ACTUAL e2_battery.py production corroboration lines but sweep N finer (2^19, 2^19.5 equivalent,
  2^20) to nail down exactly where the sign flips, and confirm this doesn't affect k=8 (larger true
  signal, more margin).
- Have not yet checked lead #3's "kernel quality 2.7e-11, sv falling to 8.2e-08, gain -1.6e-5" framing
  beyond what's covered above (I believe this is answered: real, confirmed by mpmath).
- Have not yet written final audit.md / manifest.json in research/results/f5_audit_fable/.

## NEXT STEP
1. Quick sweep: production_min_eps2-equivalent (using common.py's actual function, grid="linspace",
   default s) for gG at N = 2^18, 2^19, 2^19.5-ish (use e.g. 2**19 + 2**18), 2^20 to find the exact N
   where sign flips from false-PASSIVE to correct-GAIN, to state the fragility margin precisely.
2. Skim C50/C52/C54 sections of ns_answer.md for any numerical claim not yet controlled.
3. Write research/results/f5_audit_fable/audit.md (per-claim verdicts C48-C54) and manifest.json.
4. Final heartbeat + done.

## Scratch scripts (reproducible, live only in session scratchpad — key ones worth re-deriving if lost)
- Independent mpmath high-precision Delta sup check (derivative-sign-scan + bisection, not
  polyroots) — confirmed k=8/k=16 witnesses to 9-10 sig figs.
- Grid-convention comparison for production_min_eps2 (linspace vs periodic vs N sweep 2048..2^25) on
  the k=16 witness at s=6e-4, x0=0.4859 — the source of the C53 mechanism refinement above.
All are short (~30-60 line) standalone scripts using only common.py's Atomic/staircase_eps1 plus
tmm_h.ht_derivative directly; can be regenerated quickly from the descriptions above if needed.
