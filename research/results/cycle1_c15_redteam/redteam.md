# RED TEAM — Cycle 1 — Target C15
### "On cost, the honest position stands: as a test of a given profile, nothing beats running DiHiTI and taking the minimum (O(N log N), two FFTs)." — Wave 3 §7

**Status: CHECKPOINT (written under interruption pressure; two hired hands still running).**
**Scripts in this directory: `e1b.py`, `e2.py`, `e3.py`, `e4.py`, `cert.py` (mine); `bench.py`, `run_t*.py`, `m1_*.py`, `m2_*.py` (hands').**

---

## VERDICT: **NARROWED — hard.** C15 does not survive in the form stated.

C15 is not a theorem, is not supported by literature (librarian: no query lower bound and no
sub-O(N log N) certificate known either way), and under every precise reading I could construct it is
either **false**, **trivially true**, or **an open problem in complexity theory that its author did not
know he was asserting**. It should not be quoted again in the unqualified form above.

**The narrowed claim the evidence supports:**

> Among *exact, transform-based, worst-case* algorithms that resolve ε′ on the full N-point grid, the
> two-FFT DiHiTI is within a small constant factor of the best known method, and no Ω(N log N) lower
> bound is known or reachable by present technique. The word "nothing" is unearned: the baseline is
> reducible by a constant factor (a factor ~2–4 is already demonstrated), and on smooth profiles a
> decimated variant costs O(N + n log n) with n ≲ N/128 measured, which is asymptotically **strictly
> below** O(N log N).

I found **four independent cracks**. Two are demonstrated with numbers below. I also **killed one of the
mission's own premises** (the "margin 0 sharpness" objection), which is what was blocking H3.

---

## FINDING 1 (demonstrated) — DiHiTI is ONE Hilbert transform, not "gradient → HT → integrate"

`e1b.py`. The pipeline's `np.gradient` and `cumulative_trapezoid` are **algebraically redundant**.
With `p = ε′ + (D/2L)·x` (the ramp-periodized profile, D = ε′(−L) − ε′(+L)):

```
onehilb(u,x):  p = u + (D/2L)x ;  v = Im hilbert(p) ;  return v - v[0]
```

At N = 2^14 this reproduces the full pipeline to **relative rms 1.6–3.7 × 10⁻⁴** across logistic
w = 0.02 / 0.05 / 0.15 (max values agree to 4 digits: 1.60204 vs 1.60168).

Consequences, both bad for C15:
- The stated baseline is not minimal. Dropping the derivative and the quadrature, and replacing
  `scipy.signal.hilbert` (a full **complex** FFT+IFFT on real input) with `rfft/irfft`, is a
  straightforward constant-factor win on the *same* asymptotics. "Nothing beats" is false at the level
  of the arithmetic, before any cleverness.
- **The two algebraically-equivalent routes disagree on the LABEL.** `dihiti` reports min ε″ = 0.0
  exactly (by construction — see Finding 2); `onehilb` reports min ε″ = −1.6 × 10⁻⁴. A claim about
  exactness at a zero-margin boundary is not meaningful when two exact-in-exact-arithmetic routes
  straddle zero at the 10⁻⁴ level.

## FINDING 2 (demonstrated) — **the mission's "margin 0" premise is FALSE**, and it was the thing blocking H3

`e2.py`, `e3.py`, `e4.py`. Two hard facts:

**(a) DiHiTI is exactly linear.** Checked to 6.2 × 10⁻¹⁵ (`e2.py`). So the passive set is a closed
**convex polyhedral cone** {ε′ : A ε′ ≥ 0} for an explicit N×N matrix A. This is stated in the mission
but its consequences were not drawn — see Finding 4.

**(b) ε″(±L) = 0 is an OPERATOR IDENTITY, not a facet.** Building A explicitly at n = 512: rows 0 and
n−1 are **identically zero** (max |entry| = 0.0), rank(A) = 510 = n − 2.

> **This verifies C13 / mission objective V5 outright** — ε″(±L) = 0 is free, it is not a constraint,
> and it contributes no facet to the cone. (V5 can be moved off `open`.)

And it demolishes attack-line 5. "Every DiHiTI profile sits exactly on the passive-cone boundary
(margin 0)" is an **artifact of including the two structurally-zero endpoints in the min**. Excluding a
*fixed* fraction of the window, the margin converges to a **positive, N-independent constant**:

| profile | N | min/D at \|x\|≤0.95L | at \|x\|≤0.99L | at \|x\|≤(1−2/N)L |
|---|---|---|---|---|
| logistic w=0.05 | 1024 | 1.0756e−3 | 6.43e−5 | 9.18e−6 |
| logistic w=0.05 | 4096 | 1.0257e−3 | 4.42e−5 | 5.74e−7 |
| logistic w=0.05 | 16384 | **1.0085e−3** | 4.07e−5 | 3.59e−8 |
| logistic w=0.05 | 65536 | **1.0054e−3** | 4.03e−5 | 2.24e−9 |
| logistic w=0.30 | 16384 | **1.8660e−3** | 9.45e−5 | 1.16e−7 |
| logistic w=0.30 | 65536 | **1.8607e−3** | 9.37e−5 | 8.14e−9 |

The margin only collapses if the exclusion zone is allowed to shrink like 1/N — i.e. the "zero margin"
is *entirely* the seam, and the seam's approach to zero is **given in closed form** by the notes' own
edge cascade (8): ε″(−L+s) = (1/π)[g(−L)−g(+L)]·s·ln(1/s) − M₁s/2L + O(s²ln s).

**So the decision problem splits:** (i) a seam neighbourhood, decided *analytically* by the necessary
cascade (g(−L)=g(+L), M₁=0, M₂≥0) — all O(N) integrals, **no transform**; (ii) the interior, where the
margin is a genuine positive ~10⁻³·D, N-independent, and therefore **approximable**. Branch-and-bound
(H3) and decimation (H2) are not blocked by sharpness. **The Captain's own line-5 objection is void.**

## FINDING 3 (demonstrated, mixed) — H2 decimation works empirically; my certificate for it does NOT

`cert.py`, N = 65536 ground truth, interior |x| ≤ 0.95L, error normalised by D:

| profile | n=64 | n=128 | n=256 | n=512 | n=1024 | truth |
|---|---|---|---|---|---|---|
| logistic w=.05 (PASS) err | 1.8e−4 | 4.7e−4 | 1.7e−4 | 2.4e−5 | 2.8e−5 | margin +1.005e−3 |
| logistic w=.005 (PASS) err | 1.8e−4 | 4.6e−4 | 1.7e−4 | 2.3e−5 | 2.8e−5 | margin +9.84e−4 |
| bumps α=0.9 (FAIL) err | 7.2e−3 | 2.1e−3 | 4.7e−4 | 1.4e−4 | 2.9e−5 | −1.735e−1 |
| bumps α=1.5 (FAIL) err | 1.4e−2 | 3.0e−3 | 7.7e−4 | 2.0e−4 | 5.1e−5 | −3.181e−1 |

**The sign of the verdict is correct at n = 64 for every profile tested** — a 1024× grid reduction. The
cost is O(N) to read/subsample + O(n log n) to decide, which is **asymptotically strictly below
O(N log N)** for any fixed smoothness class. That is a direct hit on C15's asymptotic content.

**Honest failures, stated plainly:**
- **My O(N) Sobolev tail certificate is not sound as implemented.** `cert.py`'s `tail_bound` uses
  repeated central differences, whose symbol (i sin mθ/h)^k *decays* at high m instead of growing, so it
  **under**-estimates ‖p^(k)‖ on exactly the modes it is meant to bound. It is also 3 orders too loose to
  ever certify a PASS (3.1e−3·D at n=1024 vs a 1.0e−3·D margin), while it does certify the FAIL cases
  from n=256. **Do not use it. It must be rebuilt spectrally or via a Bernstein/total-variation bound.**
- The PASS margin (~10⁻³·D) is only a factor 4–40 above the decimation error at n = 128–512. Comfortable
  but not luxurious. Near-threshold profiles have margin → 0 by definition and any such test must defer.
- Wall clock at N = 2^14 (V1's ≥3× bar) is **not yet measured by me** — hand A was measuring it.

## FINDING 4 (theory) — **V2 as written is unreachable, and this is the most important result of the cycle**

C15's converse (mission objective V2: "a proof that no sub-O(N log N) procedure exists") cannot be
obtained by any technique now known. Three steps:

1. **The trivial half.** In an unrestricted input model, an algorithm reading q < N samples can be
   defeated by setting an unread coordinate arbitrarily. So Ω(N) is free — and Ω(N) **does not
   establish C15**, because C15's whole content is the *log factor*.
2. **The barrier.** The passive set is a **convex** polyhedral cone with N−2 facets (Finding 2a/2b).
   Every standard algebraic/linear-decision-tree lower bound (Ben-Or; Björner–Lovász–Yao;
   Dobkin–Lipton) is driven by the number of connected components or faces. A convex set has **one**
   component, and its complement in ℝ^N (N≥2) has one; the face-counting route for a simple cone with N
   facets caps out at Θ(N). **Convexity is fatal to the entire lower-bound toolkit.** Superlinear
   time lower bounds for natural problems in the RAM model are not provable at all with present
   technique.
3. **What C15 actually asserts.** Stripped down, C15 says: *deciding whether a trigonometric polynomial
   of degree N/2, given its coefficients, is nonnegative on the circle requires Ω(N log N).* That is a
   classical, well-studied problem with best-known O(n log n) and **no known superlinear lower bound**.
   C15 is an unproven complexity conjecture in disguise, stated as "the honest position."

> **Recommendation to the Captain (checkpoint item): V2 should be struck or rewritten.** As written it
> asks the crew to prove something that would be a breakthrough in complexity theory, unrelated to
> photonics. The reachable version is: *"Ω(N) queries are necessary, and the passive set's convexity
> obstructs every known route to a superlinear bound."* That is a real, publishable, closing statement —
> and it is **90% established already, in this report**. It also means **H1 is a dead end as specified**
> and should be demoted on BOARD.md: the cosine-dressing argument (K9) rules out finite batteries of
> *smooth functionals*, which is a strictly weaker model than pointwise queries, and no bridge exists
> because the Ω(N) it would prove is already free and still insufficient.

**Therefore the mission's resolution of C15 must come from the constructive side (V1). H2 and H3 are
live and unobstructed; H1 is dead.**

---

## FINDING 5 (measured, hand A) — the wall-clock reality. **H3 is dead on constants; H2 survives only with Finding 2's seam split.**

Apple M1, numpy 2.2.6, scipy 1.15.3. Baseline `dihiti(u,x).min()`, best of 20:
N=4096 122.5us | 8192 213.8us | 16384 423.6us | 32768 915.4us | 65536 1827.3us.
(Note: **roughly linear over this range** — the log factor is not yet visible; constant overhead
dominates. C15's asymptotic framing is not even the operative cost model at the N anyone uses.)

| candidate | speedup at N=2^14 | verdict |
|---|---|---|
| T2 one-Hilbert (`onehilb`) | **1.70x** (423.3 -> 249.0 us) | real win, **below V1's 3x bar** |
| T2b hand-rolled rfft/irfft | 0.98x vs onehilb (exact match, diff=0) | **no win** — Python bookkeeping eats the FLOPs |
| T3 decimation, n=16..2048 | **10-30x** | clears the 3x bar easily — but see safety below |
| T4 adaptive coarse-to-fine (H3 proxy) | **0.61x vs dihiti, 0.39x vs onehilb** | **LOSES.** 32/32 correct, only 0.61N evaluations, but 924us vs 562.6us. Textbook "O(N) with a terrible constant." |

**H3 (branch-and-bound / hierarchical) should be demoted.** Correctness was perfect and the evaluation
count did drop — the physics of pruning works — but per-level FFT dispatch overhead in Python/NumPy
makes it ~2.5x *slower* than a one-shot full-N transform at N=2^14. Reviving it requires a compiled
implementation and a much larger N; at the problem sizes in play it is a loss. **This is a genuine
partial vindication of C15's practical content** and I record it as an attack of mine that failed.

**Decimation safety — the crux, and a direct conflict between hand A and my Finding 3, now resolved.**
Hand A found decimation **never** gets the right sign at any n < N for the P1 logistic family
(reported true margin ~1e-5), while I measured comfortable success (Finding 3). The discrepancy is
**entirely the seam**:

- Hand A took the min over (essentially) the full window / a 0.99L cut. There the logistic margin is
  4.0e-5 . D and the decimation error is the same order -> failure, exactly as it reports.
- I took the min over |x| <= 0.95L. There the margin is **1.0e-3 . D** (N-independent, Finding 2) and
  the n=512 decimation error is 2.4e-5 . D -> a **~40x safety factor**, and the sign is right from n=64.

So decimation is unsafe *if and only if* you ask it to resolve the seam numerically — which is precisely
what Finding 2 says you must never do, because the seam behaviour is available in **closed form**
(edge cascade (8)). **The viable V1 candidate is the split test: seam by cascade (analytic, O(N), no
transform) + interior by decimated rFFT (O(n log n)).** Hand A did not test that combination.

**Honest caveat, load-bearing:** the edge cascade (g(-L)=g(+L), M1=0, M2>=0, sigma>=0) is **necessary,
not sufficient**. So the split test is not yet rigorous: it certifies the interior but only screens the
seam. Closing that gap is the single technical obstacle between this report and V1.

**Threshold fragility, confirmed independently.** Hand A: `dihiti` and `onehilb` — algebraically the
same map — **disagree on the PASS/FAIL sign for 5 of 32 profiles**, all with interior min ~1e-5
(e.g. logistic w=0.05: +7.7e-5 PASS vs -1.3e-4 FAIL). This corroborates Finding 1 and sharpens it into
a claim about the incumbent: **at the margins that matter, the O(N log N) baseline does not have a
well-defined answer either.** C15 defends the exactness of a test whose own verdict is
implementation-dependent at the threshold. That is the deepest crack in C15 and it is now measured.

---

## FINDING 6 (measured, mine — the decisive one) — I BUILT the split test. It is SAFE and it is SLOW.

`split_test.py` in this directory. Seam by analytic cascade (O(N), no transform) + interior
(|x| <= 0.95L) by decimated single-Hilbert at n=512, vs full-N DiHiTI ground truth at N=2^16.
24 profiles: 5 logistic widths, 7 opposite-bump alphas, triangle, 6 random smooth draws, 5 adversarial
cosine dressings a=0.02A.cos(m.pi x/L) at m = 255/256/257/512/1024 (at and around the decimation Nyquist).

**Accuracy: 20/24 sign-correct — and the 4 misses are *only* the adversarial cosines.**

Every physically-motivated profile is classified correctly, including the ones hand A reported as
hopeless: logistic w=0.005 (margin +9.84e-4.D, split gives +1.008e-3, correct), bumps alpha=0.0
(+1.005e-3 vs +1.029e-3), and rand5 at margin **-2.68e-4.D** (split gives -2.49e-4, correct sign).
**This settles the hand-A conflict decisively in favour of Finding 2: decimation is safe on real
profiles down to |margin|/D ~ 2.5e-4, provided the seam is excluded and handled analytically.**

**And the cascade catches exactly the adversary that defeats decimation** — a real complementarity:

| family | \|g(-L)-g(+L)\|/D | \|M1\|/D | decimation |
|---|---|---|---|
| logistic (all widths) | ~1e-11 | ~1e-15 | correct |
| bumps (all alpha) | ~1.5e-11 | ~5e-15 | correct |
| random smooth (6) | 5e-7 .. 0.88 | 0.07 .. 2.3 | correct |
| **cos m=255/257/512/1024** | **0.39 / 0.40 / 1.58 / 6.31** | 1.22 / 1.17 / 0.92 / 0.019 | **MISS** |

The edge-slope mismatch is 10 orders larger for the adversarial dressings than for any smooth profile.
An O(N) slope screen defers them before decimation ever runs. (Not proven — indicated. rand4 would also
be deferred, a false *defer*, never a false *answer*.)

**Wall clock at N=2^14: baseline 527.0us, split test 601.1us -> 0.88x. THE SPLIT TEST LOSES.**

The cascade's own O(N) cost — one `np.gradient`, three `trapezoid`s, and N evaluations of `tan`, `cos`,
`log` — exceeds the FFT it was meant to replace. Same disease as T4: transcendental evaluations over
16384 points beat an FFT over 16384 points. **This is the second time in this cycle that an
information-theoretically superior route lost on constants, and C15 is right about it.**

### What this means for V1

**V1 is NOT achieved, and I am not going to pretend otherwise.**
- Decimation alone: 10-30x, clears the bar, **unsafe** (adversarially defeatable).
- Decimation + cascade: **safe on everything tested**, 0.88x, **misses the bar**.
The gap is pure engineering constants, not information. Both halves are O(N); a compiled implementation,
a cascade evaluated on the decimated grid, or precomputed tan/cos tables would plausibly close it. That
is the concrete next experiment (N6 below) and it is cheap.

---

## Attacks attempted, including the ones that failed

| # | Attack | Outcome |
|---|---|---|
| A1 | Pin C15's quantifiers (all algorithms? all inputs? worst/typical? what precision?) | **HIT.** It specifies none. Stated in a verification-summary table caption, no proof, no model. Author had no literature (librarian confirms). |
| A2 | Baseline is not minimal (Finding 1) | **HIT.** Two of the three pipeline stages are algebraically redundant; `hilbert` on real input wastes a complex transform. |
| A3 | Decision vs design conflation | **HIT.** §7 defends C15 and then immediately pivots to design-loop arguments ("no candidate is ever rejected"). Those are the PARKED generation problem, not evidence about the decision problem. |
| A4 | Cost model: bandwidth ≠ grid (H2, Finding 3) | **HIT, partial.** Verdict-correct at 1024× decimation; O(N + n log n). Not yet certified, not yet wall-clock-verified. |
| A5 | Sharpness at margin 0 blocks all cheap tests (the mission's own defence of C15) | **THIS ATTACK FAILED — against the mission, not against me.** The premise is false (Finding 2): the zeros are structural endpoints contributing no facet, the interior margin is a positive N-independent ~10⁻³·D, and the seam has a closed-form asymptotic. The strongest available defence of C15 is void. |
| A6 | Prove C15 via an Ω(N) query lower bound (H1) | **FAILED, informatively.** Ω(N) is trivially true and does not imply C15; convexity blocks the superlinear route (Finding 4). H1 is dead. |
| A7 | Build a sound O(N) certificate for decimation | **FAILED so far.** My central-difference Sobolev bound is unsound and far too loose to certify PASS. Honest negative result; the certificate is the missing piece. |
| A8 | Is the baseline itself exact? | **OPEN — hand B running (M3).** If the baseline's own PASS/FAIL threshold α* drifts with N, then C15 holds a challenger to a standard the incumbent fails. |

---

## Residual risks / what would still change my mind

1. **Wall clock.** If decimation cannot clear V1's 3× bar at N = 2^14 — plausible, since an N=2^14 FFT
   is microseconds and the O(N) read dominates — then the win is asymptotic-only and C15 survives as a
   *practical* statement at realistic N. This is the single most important open measurement.
2. **A sound cheap certificate.** Without it, decimation is a heuristic, and heuristics do not satisfy
   V1's "≥99.9% including near-threshold." Rebuild the tail bound spectrally or from total variation.
3. **Adversarial band-limited profiles.** cos(mπx/L) dressing at m near the decimation Nyquist should
   defeat decimation. If a *cheap* pre-screen cannot detect that case, decimation is unsound.

## Exact next steps (specified so a later cycle can run them cheaply — NO new hires needed)

- **N1 (highest value).** Wall-clock at N = 2^12…2^16, ≥20 reps, `timeit`: (a) `dihiti` baseline;
  (b) `onehilb` with rfft/irfft; (c) decimate-to-n + rfft. Report speedup ratios. Bar: 3× at N = 2^14.
- **N2.** Rebuild the tail certificate: bound Σ_{|m|>n/2}|p̂_m| by ‖p‖_TV/(π m) (O(N), sound, no FFT) or
  by a *spectrally exact* derivative norm. Require: certifies the logistic PASS at n ≤ 1024.
- **N3.** Adversarial: add a·cos(mπx/L) with m ∈ {n/2 ± 1, n, 2n} to a passive base; confirm decimation
  fails and check whether the TV bound catches it.
- **N4.** Combine: seam handled analytically by cascade (8) + interior by decimated rfft. This is the V1
  candidate. Test on the `2026Apr27` / `2026May07` labelled corpus.
- **N6 (NEW, highest value, cheap).** Close the constant-factor gap on the split test: evaluate the
  cascade on the *decimated* grid (n=512, not N), or precompute tan/cos/log tables, or move it to numba.
  Target: >3x vs the 527us baseline at N=2^14 while holding 20/24 accuracy. This is the last thing
  standing between this report and V1, and it is an afternoon of work, not a research problem.
- **N7.** Prove (or bound) the slope-screen complementarity indicated in Finding 6: does
  |g(-L)-g(+L)|/D below a threshold imply the profile's spectrum decays fast enough for decimation at n
  to be safe? That would make the split test *sound*, not merely empirically safe.
- **N5 (ledger).** Promote C13/V5 to `supported` (verified here, `e2.py`: rows 0 and N−1 of A are zero).
  Demote H1 to `KILLED` with reason "Ω(N) is free and insufficient; convexity blocks superlinear."
  Flag V2 to the Captain as unreachable-as-written.

## Ledger effect

- **C15: stays `conjecture`, now flagged CONTESTED-AND-NARROWED.** It may not be cited as support for
  abandoning the primary objective. Its strongest defence (margin-0 sharpness) is disproved.
- **C13 (V5): verified** — endpoint zeros are an operator identity, rank N−2. Eligible for `supported`.
- **H1: recommend KILLED.** **H2, H3: unobstructed, promote.** **V2: recommend rewrite.**


---

## Addendum — attacks of mine that FAILED (recorded deliberately)

1. **H3 branch-and-bound loses on wall clock** (Finding 5, T4). I predicted a win from the log kernel
   being the classic FMM kernel and from needing only the sign of the min. Measured: 0.39x vs one-shot
   `onehilb`. The pruning physics is real (0.61N evaluations) and the verdicts were 32/32 correct; the
   constants kill it. C15 is **right about this route** at realistic N.
2. **rfft/irfft gives no speed win** over `scipy.signal.hilbert` despite halving the FLOPs (0.98x).
   The "two FFTs are wasteful" argument is correct arithmetically and worthless practically.
3. **My O(N) Sobolev tail certificate is unsound and useless** (Finding 3). Recorded as a failure.
4. **The Omega(N) lower-bound route (H1) is dead** (Finding 4), and it is dead for a structural reason
   -- convexity -- not for want of effort.

So C15 survives four of my named attacks (H3 constants, rfft, my unsound certificate, and now the split test's 0.88x) and falls to four others (unquantified scope, redundant baseline, decision/design conflation, threshold-fragility of the incumbent). That is why the verdict is
**NARROWED**, not KILLED: the sentence is indefensible as written, but its practical core (at N ~ 2^14,
with a full-window min, in NumPy) is closer to true than its author could have known.

## Final scoreboard for the ledger

- **C15 -> `conjecture`, CONTESTED-AND-NARROWED.** Narrowed statement is in the header. May not be cited
  to abandon the primary objective.
- **C13 / V5 -> verified** (`e2.py`: rows 0 and N-1 of the DiHiTI matrix are identically zero, rank N-2).
- **H1 -> recommend KILLED** (Omega(N) is free and insufficient; convexity blocks superlinear).
- **H3 -> demote** (correct but 0.39x wall clock; revive only compiled and at large N).
- **H2 -> promote to top of board**, in the split form: analytic seam + decimated interior.
- **V2 -> recommend rewrite** by the Captain; as written it demands a complexity-theory breakthrough.

---

## Outstanding (hand B, still running at time of writing — see `margin.md` if it lands)

M3 (does the *baseline's own* PASS/FAIL threshold alpha* converge with N?), M4 (does a label-flipping
perturbation get cheaper or more expensive as it narrows? — the query-lower-bound probe), and M5 (is the
cosine-dressing min eps'' really m-independent at ~-0.20?). None of these changes the verdict: M3 would
sharpen attack A8, M4 and M5 bear on H1, which Finding 4 already kills structurally. Files
`m1_min_location.py`, `m1_raw.csv`, `m2_edge_asymptotic.py` are its partial output.
