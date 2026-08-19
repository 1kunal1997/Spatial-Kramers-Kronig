# Ship's Log

Append-only. Newest at the bottom. One entry per cycle, however brief.
Negative results get logged with the same care as positive ones.

---

## Cycle 0 — 2026-08-18 — Fitting out

Crew quarters built before the mission brief arrived. Adversarial crew (`red-team`, `replicator`,
`auditor`, `referee`) hired and stationed — these are problem-independent. Ship's Log and
`PROTOCOL.md` laid down. Line crew roster deferred until the brief so it can be tailored.

Status: **awaiting Captain's brief in `MISSION.md`.** Not yet sailing.

---

## Cycle 0.5 — 2026-08-18 — Briefing received

Captain's brief loaded. `MISSION.md` written: **cheap certificates for DiHiTI passivity**. Three
waves of prior work (Apr–May trial-and-error; Aug-13 potential theory; Aug-18 Fable notes) read in
full and decomposed into 17 inherited claims (`CLAIMS.md`, all at `conjecture`), 12 dead ends
(`KILLED.md`), and 8 hypotheses (`BOARD.md`).

**Central finding of the read:** the primary objective — a test cheaper than the two-FFT
O(N log N) baseline — is **open**, and the only thing standing where an answer should be is C15,
an *assertion* in Wave 3's verification summary that "nothing beats running DiHiTI." The same
species of assertion was broken three-out-of-four times on Aug 13. C15 is the mission's first
target, and both outcomes (cheap test found / lower bound proved) are wins.

**Reframing proposed to the Captain:** the primary objective is a **computational complexity**
question, and every wave so far has attacked it as a function-theory question. That reframing
generates H1 (Ω(N) adversary lower bound, reusing the Aug-13 cosine-dressing construction, which is
already most of a proof), H2 (cost in effective bandwidth n, not grid N), H3 (branch-and-bound on
the log potential — ln is the classic FMM kernel), and H4 (Chebyshev-system collapse of the infinite
LP to finitely many nodes).

The flipped optimization work is parked in `PARKED.md` with rules of engagement, not deleted.

Status: **awaiting Captain's A/B/C decision and crew-roster confirmation before hiring line crew.**

---

## Cycle 0.6 — 2026-08-18 — Parsimony made binding

Captain's structural critique of the prior corpus recorded as a **binding victory condition (V6)**,
not a stylistic note: the final characterization must be **necessary and sufficient** and consist of
**one or two** conditions, with existing results recovered as corollaries.

The knock, made concrete: the corpus holds ~12 overlapping statements, and exactly **one** (Thm 3.1)
is two-sided — the one the Captain rejected as a re-representation of the transform. Everything else
is one-sided. Wave 3's own admission that Thms 4.2 and 5.2 are "genuinely complementary: neither
contains the other" is the clearest symptom: two sufficient conditions, neither implying the other,
both required for coverage.

**H9 added and put at top priority** — unification via potential domination. Thm 3.1(iii) already
reads as a single law (the drop measure's log potential is dominated by that of the point mass at i,
a balayage/majorization order); middle-half and unimodality then become two sufficient conditions for
**one** relation rather than sibling theorems. It is the only hypothesis on the board that attacks
the scatter instead of adding to it.

`referee` charter amended: sprawl is now an explicit, valid ground for rejection. A true, novel,
one-sided thirteenth theorem gets rejected on parsimony.

---

## Cycle 1 — 2026-08-18 — SAILING. Four hands dispatched in parallel.

Line crew hired (`.claude/agents/`: theorist, numericist, analyst, librarian + the four adversaries).
Note: newly written agent definitions are not registered until a session restart, so cycle 1 was
dispatched as general-purpose hands each pointed at its own charter file. Future sessions — including
cron-spawned ones — will address the crew by name.

Captain added (2026-08-18): `theory/figures/2026Apr27/` (73 figs) and `theory/figures/2026May07/`
(25 figs) are an **abundant labelled example corpus**, not just the seed figure. Recorded in
MISSION.md as the mission's labelled test set and passed to every hand. Wave 1 produced no
conditions, but it produced *data* — that data is now an asset for validating any proposed criterion
and for sourcing near-threshold adversarial cases.

**Dispatched:**

| Hand | Target | Why |
|---|---|---|
| theorist (opus/xhigh) | **H9** — unification via potential domination | Top priority. Only hypothesis attacking the *scatter* (V6) rather than adding to it. Key test: do Thm 4.2 and Thm 5.2 both fall out of one domination relation as corollaries? |
| numericist (sonnet) | **H6** + **H7** | H6 = the mission seed (V4): does the Clausen dip tolerance d*(b) predict the Captain's α crossover in (0.6, 1.0)? H7 = verify ε″(±L)=0 is automatic (V5), cheap. |
| red-team (opus/xhigh) | **C15** | Attack "nothing beats running DiHiTI" from both sides. Break it → V1; prove it → V2. Guard against the third outcome: accepting it on the author's confidence. |
| librarian (sonnet) | prior art | Potential domination / balayage; Chebyshev-system collapse of semi-infinite LPs; query lower bounds for non-negativity testing + FMM for log kernels. Stop the crew re-deriving classical results. |

Scaffolding committed (8522a35) before sailing.

### Cycle 1 addendum — engine reality check

Set up the Captain's standing order: a **status summary every 5 hours** (recurring job, fires at :23,
auto-expires after 7 days), which also integrates returned crew findings and dispatches the next
cycle rather than merely reporting.

**Correction to the approved plan.** Layer 2 ("cron resurrection surviving session death") does not
work as I described it. Local cron jobs here are **session-only** — nothing is written to disk and
they die with the session. The `durable` flag has no effect. Cloud routines *are* genuinely durable,
but they run on a **GitHub checkout in Anthropic's cloud**, not the laptop, and two things block that
path today:

1. GitHub App access to `1kunal1997/Spatial-Kramers-Kronig` could not be verified.
2. **Key mission inputs are untracked**: `verify_passivity.py`, the Aug-13/Aug-18 PDF notes, and the
   entire labelled figure corpus — `*.pdf` and `*.png` are gitignored by project policy. A cloud
   agent would clone a repo missing most of the mission's evidence.

What survives unchanged: **the ledger is the durability mechanism.** `research/` is committed, and
any new session resumes the voyage cold from `MISSION.md` + `PROTOCOL.md`. What is lost is only the
*automatic* restart with the laptop closed. Both items queued for the Captain; neither blocks work.

---

## Cycle 1 — librarian returned

Report: `research/lit/cycle1_domination_and_complexity.md`. Three findings, one of them a live lead.

**1. The obvious classical match is a FALSE FRIEND — recorded as a trap.** The crew's inequality does
have classical names attached: **Cartan's domination principle** (Brélot 1959) and **balayage**. But
both are *one-way sufficiency* statements ("true on the support ⟹ true everywhere") — **not** the
necessary-and-sufficient comparison the mission requires. Adjacent vocabulary, not a solved instance.
If a later cycle cites Cartan/balayage as closing V1 or V3, that is an error; flagged for `red-team`.

**2. The real lead: Cartier–Fell–Meyer / the Choquet convex (dilation) order.** An N&S theorem for
exactly this *shape* of comparison — one measure's potential dominating another's, tested against a
family of convex/superharmonic functions — with the 1-D case going back to **Hardy–Littlewood–Pólya
majorization**. The librarian found **nobody** who has specialized it to "log-potential of an interval
measure vs. a point mass at i." That specialization is the open step, and it is precisely H9's target.
**Forwarded to the theorist mid-flight.**

**3. Front 3 is genuinely open territory.** No query-complexity lower bound exists for this
non-negativity-testing problem; no algorithm known for finding a log potential's minimum cheaper than
the full transform; no sub-O(N log N) Toeplitz PSD certificate. Cuts both ways, honestly: good for V2
(nobody has beaten us to the impossibility proof) and bad for V1 (no literature shortcut to the
speedup). Notably it also means **C15 is not supported by any known result** — its author had no
literature to lean on either.

Front 2: Charnes–Cooper–Kortanek (1963) semi-infinite-LP-via-Carathéodory answers a *different*
question (structure of an optimum, not feasibility of a candidate) — do not treat it as solving V1.
Confirmed (not broken) that Carathéodory–Toeplitz has no known finite truncation shortcut, consistent
with K2.

**Access caveat, recorded for the gate:** Saff–Totik and Landkof were paywalled and not fetchable in
full text. Anything attributed to them is *existence-only*, unverified against primary theorem
numbers. Any claim resting on them cannot pass `referee` until checked against the primary source.

Nothing added to `KILLED.md` — no hypothesis died this pass.

---

## Cycle 1 — numericist returned. V5 supported; V4 answered numerically with a twist.

**H7 / C13 → `supported`. V5 holds and drops out of the mission as a constraint.** 6 profile families
(logistic ×4 widths, plateau, off-centre step, and the α = 1.5 *gain-carrying* case) × 6 grids
(N = 2001–64001). De-tilted endpoints are exactly 0.0 by construction. The raw pre-de-tilt mismatch
sits at float64 noise (~1e−15) for smooth profiles and decays O(1/N) for profiles with sharp interior
features — pure quadrature/Gibbs. The de-tilt is genuinely a no-op, **even for a non-passive input**.
No family broke it.

**H6 / V4 — the crossover is quantified, and the Captain's diagnostic was hiding the pattern.**

Source located and confirmed (`theory/mismatch_table.py:641-677`); the Captain's exact numbers
reproduced (α = 1.0 → −0.1168, α = 1.5 → −0.3017), so the family and parameters are verified, not
reconstructed.

**True grid-converged crossover: α_c = 0.66623** (bisected on the *interior* minimum; stable to
< 1e−5 over N = 10⁴–1.6×10⁵). That is a direct quantitative answer to V4, obtained **independently of
the Clausen machinery** — it does not depend on the unverified Wave-3 apparatus.

**C19 — the methodological finding, and the likely reason the pattern resisted quantification.**
`min ε″` taken over the *whole domain* reads **exactly 0.000 for every passive profile**, no matter
how close to failure, because the minimum is attained at the boundary — which C13 has just shown is
pinned to zero by construction. The interior dip is still positive and only crosses later. So the
diagnostic used throughout Wave 1 offers **no margin and no early warning**: it is flat at 0.000 and
then jumps to −0.117. The Captain's (0.6, 1.0) bracket was not coarse sampling — his instrument had
no resolution to give. *The informative quantity is the interior minimum.*

**Consequence to check next cycle:** every PASS label in the 98-figure corpus was produced by this
blind indicator. The labels should still be *correct* (min = 0 ⟺ passive), but every margin is
invisible, and any near-threshold case is indistinguishable from a comfortably-passive one. That
matters for using the corpus as a test set.

**C20 — the closed form does not apply as stated.** g for this family is not §5.1's rectangular
"plateau minus centred dip": it has three signed regions (elevated off-centre-peaked core, a genuine
negative annular dip near s ≈ 1.07, decay to ~0 at the edge). So the narrow d*(b) hypotheses are not
met. **The H6 hypothesis as posed is therefore not yet confirmed or refuted** — the general signed
layer-cake/Clausen reconstruction was implemented (`clausen_general_form.py`) but killed mid-run by
the checkpoint order, and was written as a slow O(N²) loop needing vectorization. Next steps are in
its `notes.md`. Note honestly: that general form is O(N), i.e. **Fallback-1 tier, not a cheap
certificate** — it would satisfy V3, not V1.

Files: `research/results/cycle1_h6_crossover/`, `research/results/cycle1_h7_endpoints/`.

---

## Cycle 1 — red-team returned. VERDICT on C15: **NARROWED**. A mission premise is dead.

Report: `results/cycle1_c15_redteam/redteam.md` (559 lines) + `bench.md`, `margin.md`, full script
suite. It hired a nested hand ("Hand B") which **contradicted it twice**; both contradictions are
recorded at the top of its own report. The adversarial architecture worked as designed — on itself.

### C15 → NARROWED (not killed, not survived)

Indefensible **as written**: an unproved assertion naming no algorithm class, input model, or
precision. But red-team built two candidate cheap tests and **both lost on wall clock**. So V1 is not
achieved either. Honest middle verdict.

**Attacks that landed:**
- DiHiTI's `gradient → HT → integrate` is algebraically **one** Hilbert transform of the
  ramp-periodized profile (rel. rms 1.6e−4). Measured **1.70×** speedup. *The baseline C15 defends is
  itself reducible* — the incumbent was never optimal.

**Attacks that failed — recorded as evidence FOR C15** (a claim surviving named attacks is evidence):
- H3 branch-and-bound: 32/32 correct, 0.61N evaluations, but **0.39×** wall clock. Constants kill it.
- rfft/irfft: exact, **0.98×**.
- Split test (analytic seam + decimated interior): 20/24 correct, all 4 misses adversarial cosine
  dressings, safe to |margin|/D = 2.5e−4 — but **0.88×**.
- ⚠ Its own O(N) Sobolev tail certificate is **UNSOUND** — central differences under-estimate the very
  modes they bound. **Do not reuse.** → `KILLED.md` K13.

### ⭐ A FOUNDATIONAL MISSION PREMISE IS KILLED

`MISSION.md` V1 asserted: *"every DiHiTI profile sits exactly on the passive cone boundary (margin 0),
so any criterion that is merely approximate misclassifies precisely the profiles of interest."*
This was inherited from Wave 2 §3 and used to argue that no approximate test can ever work.

**It is an artifact.** DiHiTI is exactly linear (6e−15), so the passive set is a convex polyhedral
cone; building the matrix explicitly, **rows 0 and N−1 are identically zero — rank N−2**. So
ε″(±L) = 0 is an *operator identity*, not a facet of the cone. The "margin 0" was measuring the
structurally-zero endpoints. **At a fixed window fraction the interior margin converges to a positive,
N-independent ~1e−3·D.**

Consequences, and they are large:
1. **The sharpness objection against approximate tests is void.** There is real margin to work with.
   This reopens the entire class of cheap approximate certificates that the premise had ruled out.
2. **C13/V5 verified outright by an independent route** — operator rank, not the numerical pipeline.
3. **C19 independently corroborated by Hand B**: 95.5% of passive profiles have their global minimum
   literally at the endpoint; interior margin/D median +1.08e−3. Two hands, two routes, same finding.

### Hand B contradicted red-team twice — recorded honestly

- The baseline's threshold **is** grid-robust (α* spread 0.056%), so red-team's "the incumbent is
  ambiguous" attack **fails** — a point *for* C15.
- Localized perturbations get **cheaper** as they narrow, so **Ω(N) is genuinely provable**; red-team
  was wrong to call H1 dead. **H1 revived.**

Red-team's core argument survives both: Ω(N) does not establish C15, whose content is the *log
factor*, and convexity blocks every known superlinear route.

### 🚩 CAPTAIN CHECKPOINT — V2 is unreachable as written

Red-team's finding: **V2 asks for a superlinear time lower bound on a problem that reduces to testing
nonnegativity of a degree-N/2 trigonometric polynomial.** That is not provable as stated. The report
claims to ~90% establish the *reachable* version. **This requires the Captain's decision — the First
Mate does not rewrite victory conditions.** Queued; V2 flagged contested in `MISSION.md`, not edited.

### Next step, specified and cheap

**E1(a): move the cascade onto the decimated grid (~20 lines, ~1 hour).** The split test is already
safe on every physical profile; its only deficiency is N transcendental evaluations. If the cascade
survives at n = 512, the test lands in **10–30× territory and C15 dies outright.** This is the single
highest-value item on the board and it is cheap.

---

## Cycle 1 — theorist returned. **H9 CONFIRMED. The unification landed. 14 statements → 3.**

Report: `theory/h9_potential_domination.md` (cold-start handoff in §12) + `h9_domination_check.py`,
`h9_fixes.py`.

### The law (L1) — proved, verified to 1.2e−4

Passivity ⟺ **Φ_P ≤ Φ_ω on ℝ**, where Φ_μ(η) = ∫ln|η−ξ|dμ and ω is the Cauchy measure. The key
identification: **ω = Bal(δ_i, ℝ) = harmonic measure of ℂ₊ at i = the push-forward of the window's own
equilibrium measure** under ξ = tan(πt/2L). *The "point mass at i" is not an extra object — it is the
seam.* Exact identity: πε″ = D(Φ_ω − Φ_P). Domination on ℝ ⟺ on ℂ by the minimum principle.

### The key test PASSED — Thm 4.2 and Thm 5.2 are ONE theorem

Both are **(L2), the extreme-point reduction** (Bauer/Choquet: 𝒦 ⊆ 𝒫 ⟺ ext(𝒦) ⊆ 𝒫) applied to two
convex families: middle-half → ≤2-atom extreme points (Richter/Winkler) → Lemma 4.1; unimodal →
plateau extreme points (**Khinchin's theorem**) → Clausen margin. **They differ only in the
decomposition chosen, not in content.** Wave 3's "genuinely complementary, neither contains the other"
is explained and dissolved: a plateau is also a mixture of symmetric pairs, and pairs with s > L/2 are
individually non-passive — convexity being one-way, that decomposition simply certifies less.

### The unification generated a sharper theorem — not cosmetic

- **One-line proof of Lemma 4.1**: reduces to A + B ≤ π/2 ⟹ sin A ≤ cos B and sin B ≤ cos A ⟹ both
  logs ≤ 0. Replaces Wave 3's AM–GM + Cauchy–Schwarz.
- **Cor 4.A (exact, two-sided)**: a balanced 2-atom drop is passive **iff t₊ + t₋ ≤ L**, closed-form
  depth. Contains Cor 4.3 and 4.4.
- ⭐ **Cor 4.B — SUPERSEDES Thm 4.2**: g ≥ 0, M₁ = 0, supp g ⊆ [−a, b] ⟹ passive **iff a + b ≤ L**.
  The middle half is merely the symmetric case; **[−L/4, 3L/4] is equally passive** — a fact the
  middle-half rule could not see. Verified: bisected threshold 1.00000L at asymmetry ratios 1/2/5;
  5×200 MC at a+b = L worst min ε″/D = 0.0; necessity matches closed form to 4 decimals.
  **This is the mission's first two-sided non-trivial claim → `replicator` next cycle.**
- C8 cascade + C9 depth law + C13 endpoint = **one Taylor expansion of (L1) at η → ∞**.

### Parsimony (V6): 14 statements → 3

(L1) the law, (L2) the reduction, (L3) node collapse. (L3) is the single net addition and is a
*reduction of* (L1), not a new sufficient condition. **Nine former theorems are now corollaries.**

### The librarian's trap was avoided — and honestly labelled

Cartan/balayage are **not** leaned on (only the harmonic minimum principle). And this is **NOT** the
full Choquet/CFM order: the test cone is only the logarithmic potentials, not all subharmonic
functions, and the harmonic Riesz term does not vanish in general. Labelled **conjecture**: since ω is
Cauchy (all moments ≥ 1 infinite), harmonic test functions with convergent ω-integral may be
essentially affine, and the affine part is killed by the necessary M₁ = 0 — if provable, CFM applies
verbatim. Not attempted; next-step #2.

### Part 3 (finite reduction) — partial, and honest about the speed

- **Proved negative**: no *profile-independent* finite node set works (5 nodes all positive, global
  min −0.0403). **This closes the naive H4** → K17.
- **Proved positive (L3)**: an n-atom drop gives Δ′ = Q/[(1+η²)∏(η−ξ)] with deg Q ≤ n ⟹ **≤ n
  candidate η — an exact, grid-free test.** Verified to **9.1e−11** against a 4×10⁵-point kernel sum
  on 300 profiles.
- **Honest negative on speed**: at n = 40 the node test is **0.2×** (slower) than DiHiTI @ 2¹⁴.
  **It does not defeat C15.** Structural win, not a speed win.
- H4's Chebyshev property is real but on the **wrong side**: {ln|ξ−η_j|} is a Chebyshev system in ξ
  (the measure side, which powers the extreme-point counts), not in η.

### ⚠ Known defect, recorded

Corpus verdict agreement **233/240**. All 7 disagreements are edge-slope-unmatched families
(g(+L) ≠ g(−L)) where the Cayley integral is only conditionally convergent and the 0.97L truncation
cuts the near-seam minimum. Diagnosed as a ξ-space **quadrature** defect, not a defect of (L1); those
profiles already fail the necessary cascade. 20-minute fix, next-step #5. **Not yet fixed — the
auditor should confirm the diagnosis rather than take it on faith.**

### Top next step

**Open Problem 8.1 (skew tolerance) is now a finite computation**: 3 linear constraints ⟹ ≤3-atom
extreme points ⟹ degree-3 Q ⟹ **Cardano**. It should close completely.

---

## Cycle 2 — 2026-08-19 09:10 UTC — EXPEDITION UNDERWAY. Autonomous overnight run.

Captain asleep, terminal open, full autonomy authorized. Parallel dispatch authorized ("work as
naturally and optimally as possible"). **V2 restated per Captain's decision (option A)** — see
MISSION.md V2a/V2b: prove the Ω(N) query bound, and characterize the log-factor barrier honestly as
*structural/conditional*, never as an unconditional lower bound.

Standing orders now in force: **no model downgrades** (`BUDGET.md`), **no task-length caps** — hands
take as long as the work needs — with exposure bounded instead by **resumable 15-minute checkpoints**
plus `hb.sh` heartbeats. Hourly cron installed (fires at :47): integrates returns, dispatches the next
cycle, commits. When the window is exhausted its fires simply fail; **the first fire after the window
resets resumes the voyage automatically.**

**Dispatched (4 in parallel):**

| Hand | Task | Why |
|---|---|---|
| theorist (opus/xhigh) | T3 fix (233/240 defect) → T1 (skew tolerance via Cardano) → T2 (CFM conjecture) | T3 is the weakest point in the strongest result and pure referee-bait. T1 should close Open Problem 8.1 outright. |
| red-team (opus/xhigh) | **E1(a)** decimated cascade → write up **V2a/V2b** | E1(a) is the highest-value item on the board: if the cascade survives at n = 512, the test reaches 10–30× and **C15 dies** → V1. |
| replicator (sonnet) | **BLIND** replication of **Cor 4.B** (C25) | The mission's first two-sided non-trivial claim. Explicitly barred from all seven originating files. |
| auditor (sonnet) | Audit C28, C18/C19, C21/C22 | C28 is a self-diagnosis by the agent whose result is at stake — precisely the situation the auditor exists for. Also checks whether K13's unsound reasoning leaked elsewhere. |

Deliberate design note: the theorist is fixing C28 *while* the auditor independently checks its
diagnosis, and neither may defer to the other. Cycle 1 showed the value of this — red-team's own
nested hand contradicted it twice, and both contradictions made it into its report.

### Cycle 2 — auditor returned: **FINDINGS**. The gate caught a bad diagnosis in flight.

Report: `results/cycle2_audit/audit.md` + independent reimplementations `audit_c28.py`, `audit_c28b.py`.
Nothing existing was modified; every check was a fresh implementation, not a re-run of the original.

**HIGH severity — C28's diagnosis is falsified (K18).** The theorist attributed (L1)'s 7/240 corpus
disagreements to a 0.97L truncation cutting the near-seam minimum on edge-slope-unmatched families,
and called it a 20-minute fix. The auditor reproduced 233/240 exactly, then tested the stated
mechanism directly:

- **Truncation is not the cause.** Swept 0.90 → 0.99999, a ~30,000× tightening. **5 of 7 profiles are
  frac-invariant to the last decimal**; the other 2 move but **plateau at a stable nonzero
  disagreement**. Tightening cannot close the gap.
- **"All 7 are edge-slope-unmatched" is false.** Only **3 of 7** have g(+L) ≠ g(−L). The other 4 —
  **both `logistic` cases, `opposite_bumps`, `lorentzian`** — match to machine precision.

**This matters more than a quadrature nuisance: those four are the project's core profiles.** The
paper's logistic and the Captain's own opposite-bumps family are exactly what (L1) exists to explain.

**Acted on immediately** — the theorist was mid-flight building the fix on the false diagnosis, and
was redirected before wasting the cycle. It was also handed a hypothesis that could exonerate (L1)
entirely: **the corpus labels may be wrong.** They were produced by the C19 *blind* indicator, which
reads exactly 0.000 for every passive profile and cannot resolve near-threshold cases — precisely
where a disagreement would appear. If (L1) is right and the labels are wrong, this is (L1) *catching
errors in the corpus*, which is a result rather than a defect. Tracked as **C29**, open.

**Everything else audited came back CLEAN**, each independently reimplemented rather than re-run:
- **C18** (α_c = 0.66623) — convergence reproduced, plus an 8× interior-window sweep *not* in the
  original notes showing it is more robust than claimed.
- **C19** (blind indicator) — confirmed.
- **C13** (endpoints) — clean O(1/N) decay, ratio 1.999–2.000 across N-doublings.
- **C21** (rank N−2, zero endpoint rows, exact linearity) — operator matrix rebuilt independently at
  several N and L; all three sub-claims hold.
- **C22** (DiHiTI ≡ one Hilbert transform) — both pipelines reimplemented; **1.58× on the auditor's
  machine** vs red-team's 1.70×. Same order, same conclusion, still **below the 3× bar for V1**.
  Recorded as an honest discrepancy in the measurement, not a contradiction.
- **K13 leak check** — the unsound Sobolev certificate is flagged everywhere it appears; no other
  finding depends on it.

Coverage stated honestly: C1–C12, C14–C17, C20, C23–C27 were out of scope this cycle and remain
unaudited.

**This is the second time in two cycles that an adversary has overturned a line-crew result** (first
red-team's nested hand contradicting red-team, now the auditor falsifying the theorist). The gate is
doing what it was built to do.

### Cycle 2 — replicator returned: **MATCH**. ⭐ First claim to clear the blind-replication rung.

**C25 (Cor 4.B) → `replicated`.** Report: `results/cycle2_corB_replication/replication.md` (+ manifest,
CHECKPOINT.md, four route scripts). Blindness held — none of the seven off-limits derivation files
were opened.

Three independent routes:

1. **Analytic (strongest).** Derived the periodic Hilbert transform of a point mass **from scratch**
   via Fourier series on the circle, sign/scale-calibrated against `scipy.signal.hilbert` (0.066% off
   unity). Closed-form two-atom ε″ swept over 29 configurations — symmetric, a = L/4 & b = 3L/4,
   strongly asymmetric, near-full-one-side. **29/29 match**, with the sign flipping at
   (a+b)/L = 1.0000 to machine precision in *every* family.
2. **Cross-check.** Own O(N²) cotangent-kernel discrete Hilbert transform vs `scipy.signal.hilbert`:
   agreement to **1e−15**, confirming `ht_derivative` uses exactly the standard periodic DHT. *This
   independently validates the project's core pipeline, which no prior wave had checked from outside.*
3. **Numeric, own construction.** 8 smooth families through production `tmm_helper.ht_derivative`.
   Sufficiency (a+b ≤ L): **104/104 pass, zero mismatches.**

**The part that shows the role working as designed.** Necessity initially looked like **76/80
mismatches** — profiles with a+b > L staying passive. The replicator **investigated rather than tuned
it away**, traced it to non-adversarial construction (effective support narrower than the nominal
box), then built a separate mass-normalized narrow-bump study with proper w → 0 grid scaling
(N up to 1.6×10⁶). min ε″ converged to route 1's exact values with correct sign in all three tested
(a,b) pairs. **Resolved legitimately, not papered over** — and it said so explicitly.

**Honest residual gap, self-reported:** the near-threshold strongly-asymmetric mass-normalized case
reached ~80% of target magnitude (correct sign) at N ≈ 1.6×10⁶ and was **still closing monotonically**
— stopped for diminishing returns, not doubt. Recorded as a stated limitation rather than a clean
pass.

**Load-bearing consequence for the voyage:** C25 was replicated **independently of (L1)** — route 1
derives everything from scratch. So even if the theorist finds a genuine defect in (L1) while chasing
C29, **Cor 4.B does not automatically fall with it.** The mission's first two-sided theorem rests on
its own foundation.

Next rungs for C25: `red-team` (→ hardened) then `referee` (→ ESTABLISHED). Queued for the next cycle;
red-team is currently on E1(a), the higher-value target.

### Cycle 2 — red-team returned. E1(a) **FAILED** (clean negative). **V2a PROVED.** C22 self-narrowed.

Files: `results/cycle2_e1a/redteam.md`, `theory/v2_lower_bound.md`, `results/cycle2_v2a/`.

**It killed its own proposal, found an error in its own cycle-1 report, and withdrew its own attack.**
That is the behaviour the seat exists for, and it cost the mission its most promising V1 lead.

#### E1(a) → KILLED (K19). Three independent causes.

1. **Cycle 1 never wired the cascade in.** `split_verdict()` returns the decimated minimum and never
   uses `cascade()`. So "20/24 correct, misses only the adversarial cosines" was the accuracy of
   **decimation alone**, and the 0.88× was paid for a screen that bought nothing.
2. **The screen dies on the decimated grid in *both directions at once*.** Edge-slope mismatch
   1.2e−11 → **2.3e−3** at n = 512 (false defers); adversary cos m=257 goes 3.98e−01 → **3.5e−9** — it
   aliases away, **reads cleaner than the logistic**, and answers with the wrong sign. Blind to exactly
   what it exists to catch.
3. **No threshold rescues it.** 230 profiles × n ∈ {128…2048} × τ ∈ {1e−5…∞}: every zero-false-answer
   configuration is **≤ 1.11×**; everything above 3× admits false answers.

Best honest number: **2.86×, below the 3× bar, with an uncertified gate. V1 not achieved.**

#### ⚠️ C22 narrowed by its own author — and the audit missed it

`np.linspace(-L,L,N)` **duplicates the endpoint**, so `hilbert` saw period 2L·N/(N−1). Fixing it makes
the identity *exact* (4 digits, better than claimed) — but N−1 = 65535 = 3·5·17·257 is FFT-hostile,
and the correct version runs at **0.44×**. **The 1.70× was an artifact of the bug**, surviving only
because N was a power of two.

**The auditor had independently reimplemented this and returned CLEAN at 1.58×.** Both wrote the same
`linspace` idiom, so both carried the same bug. **Independent reimplementation is not immunity when
implementers share a language, a library, and an obvious idiom.** Recorded as a standing caution in
`PROTOCOL.md`.

Consequently **cycle-1 attack 2.2 is WITHDRAWN (K20)**. With Hand B having already withdrawn the other
crack, **both of cycle 1's live cracks against C15 are gone.** C15's practical core has now defeated
**seven named challenger routes.**

#### ⭐⭐⭐ V2a is PROVED — query complexity is Θ(N), tight (C30)

Not merely formalized. The key step is a clean local law: **a bump of amplitude a and *any* width δ
depresses ε″ by −a/(πC_φ), independent of δ** — because the singular part of the DiHiTI kernel is the
Hilbert kernel, and ℋ commutes with dilation. So a_min(x₀) = κ·ε″₀(x₀) with κ/π = 0.418, constant to
±3% while ε″₀ varies over **three decades**.

That yields Θ(N) disjoint compact supports with a **uniform, N-independent** amplitude bound
(r/D = 1.680, identical to four digits at N = 4096/8192/16384; 41/41 positions flip) → pigeonhole →
Ω(N). The bound holds against the **finite** (M+1)-input family, which forecloses the "permissive
input class" objection; Yao's principle extends it to randomized algorithms. Since the baseline reads
N, **query complexity is Θ(N) — tight.**

#### V2b written as a strict conditional (C31) — sharper than cycle 1's version

Cycle 1 said "convexity is fatal to the toolkit." The written version is precise: **both known
superlinear techniques are entropy counts of combinatorial multiplicity** — Ben-Or counts components,
Björner–Lovász–Yao counts faces — and a convex polyhedral cone with Θ(N) facets has **1 component** and
≤2^Θ(N) faces, so both are capped at O(N) **by arithmetic**. Element distinctness draws its N log N
from N! orderings; there is no analogue here. Morgenstern's Ω(N log N) concerns *computing* the DFT in
bounded-coefficient linear circuits and does not transfer to deciding one bit.

Carries a scope statement forbidding the unconditional reading, an explicit non-claims section, and
three named escape routes — **R3 (bounded-coefficient linear decision trees) flagged as the
crew-sized one.**

§3.5 also explains E1(a)'s failure abstractly: a bandwidth promise defeats the lower bound, but
*verifying* one is itself Ω(N), and bounding between-sample behaviour from samples is impossible
without an a priori class. **That is why every certificate came out unsound** — H2 demoted accordingly.

#### The narrative that is emerging

Two cycles, seven challenger routes, all failed; the query complexity is now **proved tight at Θ(N)**;
and the log factor is blocked by a method-relative barrier stated honestly as conditional. The honest
headline is *"the transform is optimal up to a log factor, and here is exactly what would have to give
for that factor to fall."* **This is V2 achieved, and V1 looking genuinely hard rather than merely
unsolved.**

**One untested lever remains on V1**: every loss across two cycles has been a **Python constant, not an
information deficit**. A compiled inner loop is the only thing never tried, and 2.86× sits just under
the bar. Logged as **V1-c**, top V1 item.

### Cycle 2 — theorist: **T3 DONE (240/240)**, T1 closed negatively. Then terminated by session limit.

**T3 → 240/240 (C33).** New evaluator `theory/t3_clausen.py`, singularity-free by construction: the
drop measure is treated as piecewise-constant on grid cells with **exact** masses
w_j = ε′(t_j) − ε′(t_{j+1}); the log kernel is integrated **exactly** over each cell (all
antiderivatives are Clausen Cl₂); and Cl₂ is evaluated exactly on the uniform grid by aliasing
Σ sin(nθ)/n² into a **trigamma-weighted DFT** — verified to 1e−15 against Catalan's constant. The
whole 240-profile corpus now runs in **0.3 s**. Pointwise vs DiHiTI: median rel err 5.3e−4.

**⚖️ The C28 dispute resolved — and neither party had it right.** The auditor was correct that the
stated *mechanism* was false (truncation irrelevant; only 3/7 edge-slope-unmatched — the theorist
independently reproduced both refutations, plus a third: the DiHiTI minimum of the 5 gain cases sits
at |x|/L = 0.80–0.91, **inside** the mask). But the theorist's *conclusion* — "quadrature defect, not a
defect of (L1)" — was **correct**, as 240/240 now demonstrates.

**The true mechanism (C34) is better than either story.** The error is an **η-independent additive DC
constant**, O(h), proportional to g(+L)+g(−L): the ratio is **−0.0331…−0.0352 across all seven**, a
universal constant across offsets spanning −0.0055 → +0.289. Root cause: at t = ±L, `tan(π/2)`
evaluates in float to **1.633e16**, so ln|η−ξ| is recorded as **37.33** instead of the correct
half-cell average **8.17**. That single node supplies **one third** of the offset; the remaining two
thirds is accumulated trapezoid error on the same log singularity. Halving h halves it (ratios 2.05
across N = 1024…16384). *A floating-point overflow at one grid node, masquerading for a full cycle as a
defect in a theorem.*

**T1 / Open Problem 8.1 → CLOSED, NEGATIVELY (C35). Skew tolerance is ZERO.** Unimodal + M₁ = 0 admits
gain for **every** mode c ≠ 0 — balance is *vacuous near the seam*, because ξ = tan(φ/2) diverges
there. Verified against DiHiTI. The theorist also **retracted its own cycle-1 plan**: it had promised
Cardano, then found that unimodal extreme points are **arcs, not atoms**, hence transcendental rather
than degree-3. Second self-retraction of the voyage.

Terminated mid-flight by the session limit at ~09:50 UTC. **The checkpoint protocol worked** — T3 and
T1 were both fully on disk with a resumable state; nothing was lost.

---

## ⏱ EXPEDITION MEASUREMENT — window exhausted 09:50 UTC, reset 12:50 UTC, **auto-resume fired**

- **START** 09:10:31Z → **last crew heartbeat** 09:49:51Z ⟹ **~40 minutes of active work**
- Then ~3 h locked out. The hourly cron retried, failed harmlessly, and **the first fire after the
  reset resumed the voyage automatically, exactly as designed.**
- **Config**: 4 hands in parallel — theorist (opus/xhigh), red-team (opus/xhigh), replicator (sonnet),
  auditor (sonnet), plus First Mate integration on opus.
- 29 heartbeats from 6 distinct agents.
- ⚠️ **Caveat**: the window was *not* fresh at START — the V2 restatement, heartbeat setup and cycle-2
  dispatch had already drawn on it. A fresh window at this configuration would plausibly give ~50–60
  min, not 40.

---

## Cycle 3 — 2026-08-19 12:5x UTC — resumed automatically after window reset. 4 hands dispatched.

The auto-resume worked exactly as designed: the hourly cron retried through the lockout, failed
harmlessly, and the first fire after the 12:50 UTC reset picked the voyage back up with no
intervention and nothing lost.

| Hand | Task | Why |
|---|---|---|
| red-team (opus/xhigh) | **Harden C25 (Cor 4.B)** | At `replicated`; needs red-team → `hardened`, then referee → **ESTABLISHED**. Leading candidate for the voyage's first established claim. Pointed at the replicator's one self-reported residual (near-threshold asymmetric case at ~80% of target) and at common-mode grid bugs. |
| auditor (sonnet) | **Audit C33 / C34 / C35** | The theorist fixed its own defect and verified its own fix. It was wrong once already about this exact gap. 240/240 on one seed is not a fix — asked to re-run at other seeds, other N, non-power-of-two. |
| replicator (sonnet) | **Blind-verify C30's local law** | The Θ(N) proof rests on the width-independence of the bump depression. If that empirical core fails, V2a's proof loses its foundation. Barred from all four originating directories. |
| numericist (sonnet) | **V1-c compiled implementation** | The last untested lever on V1. Seven routes have died on Python constants, not information deficits; 2.86× sits just under the 3× bar. Explicitly warned about the `linspace` endpoint bug that produced the bogus 1.70×. |

### Cycle 3 tick — no-op (35 s after dispatch)

Queued cron fires from the ~3 h lockout are releasing back-to-back, so this heartbeat landed 35
seconds after the cycle-3 dispatch. All four hands are still working; nothing has landed on disk.
**No dispatch** — adding hands now would duplicate work already in flight.

*Operational note for future sessions:* cron fires that fail during a lockout appear to queue and
then release in rapid succession once the window resets. Expect a burst of near-simultaneous
heartbeats after any lockout, and treat all but the first as no-ops rather than as cues to dispatch.

### Cycle 3 — numericist returned. **V1-c CLOSED: clean negative with a mechanism. V1 is now answered.**

Files: `results/cycle3_v1c_compiled/` — `dihiti_fast.c`/`libdihiti.so` (ctypes C extension, `cc -O3
-march=native`, clang/arm64; no numba or Cython installed), `dihiti_fast.py`, `bench.py`,
`amdahl_diagnostic.py`, manifest + notes.

**The hypothesis was RIGHT about the part it named, and that is what kills it.** Red-team's closing
line was "every loss has been a Python constant, not an information deficit." True — for the O(N)
legs. Fused C kernels (2 passes, zero numpy temporaries, vs numpy's ~4 allocating passes) deliver
**3.34–3.65×** there.

**But those legs are not where the time goes.** The Amdahl decomposition: `scipy.signal.hilbert`
alone is **52% of baseline wall clock at N = 2¹⁴, rising to 61% at N = 65536**. So the ceiling — the
best speedup obtainable if the O(N) legs were **free** — is **already below 3× for every N ≥ 16384**,
the mission's own target N included. Measured end-to-end: **1.50×**.

Beating the FFT itself is the only lever left, and it was already dead: cycle 1's `onehilb_rfft`
reached ~1.0× once full analytic-signal reconstruction was counted.

Correctness **327/327 verdict match**, including **147/147 near-threshold** cases and the
`opposite_bumps` crossover at multiple N and α. Baseline was the honest one — identical arithmetic to
`tmm_helper.ht_derivative`, not a strawman. Grid convention chosen deliberately per the C22/K19
warning, and since the candidate reimplements identical arithmetic rather than exploiting periodicity,
the endpoint-duplication bug cannot enter; correctness was checked pointwise regardless.

---

## 🚩 SYNTHESIS — V1 is answered, and the answer is NO. Three independent levels.

With V1-c closed, the mission's primary objective has a rigorous negative answer, established at three
distinct levels rather than by exhaustion alone:

1. **Information (proved).** Query complexity is **Θ(N)** — tight (C30). Any correct test must read
   Ω(N) samples. No asymptotic shortcut exists.
2. **Method (conditional barrier).** The remaining log factor is blocked because both known
   superlinear techniques are entropy counts of combinatorial multiplicity, and a convex polyhedral
   cone caps both at O(N) *by arithmetic* (C31). Stated honestly as method-relative, **not**
   unconditional.
3. **Implementation (measured).** The constant-factor route is capped **below 3× by Amdahl** for every
   N ≥ 16384, because the FFT dominates and is itself irreducible (C36).

**Scope stated honestly:** level 3 bounds any approach that *retains* an FFT. A genuinely FFT-free
O(N) algorithm with good constants is not excluded in principle — but every such route attempted has
failed (K14 branch-and-bound 0.39×, K19 decimation, K21 certificates), and **eight named challenger
routes have now died across three cycles.**

**The Captain's bonus question has an answer:** *no, there is no cheaper test than computing the
DiHiTI — and here is precisely why, at the information, method, and implementation levels.* That is a
publishable negative result, and a considerably stronger one than the assertion C15 started as.

### Cycle 3 — navigator dispatched (strategic inflection)

V1's board items are exhausted and the mission's objectives *appear* largely resolved. That is exactly
the condition where declaring victory early is the live risk — so the navigator was spun up and
pointed **first** at the question "is the mission actually complete, or has the crew been generous
with itself?"

It was explicitly instructed to check each victory condition against the **ladder rung** rather than
against the crew's prose, and told to **contradict the First Mate's synthesis in this log if it
disagrees** — that synthesis is the most likely place premature victory has crept in, and it was
written by the agent dispatching it.

Also asked: the critical path to a defensible ESTABLISHED result; what remains genuinely open (T2/CFM,
R3, and whether an FFT-free O(N) algorithm is possible in principle — level 3 of the V1 answer only
bounds approaches that *retain* an FFT); what the Captain should be told; and whether the natural next
voyage is the parked flipped-optimization problem or something else.

### Cycle 3 — replicator returned: 🚨 **MISMATCH on C30. V2a's empirical core is contested. C30 DEMOTED.**

Report: `results/cycle3_v2a_replication/replication.md` (+ checkpoint, 4 scripts, raw JSON).

**This lands directly on the claim the First Mate reported to the Captain as "proved and tight" one
cycle ago. That report was premature and is hereby corrected.**

**The crux.** Under the literal reading — a bump added at x₀, symmetric about x₀, depression measured
at that same x₀ — width-independence is **provably false**, not merely numerically absent. The
periodic Hilbert transform maps even functions to **odd** functions about their own centre, so
ℋ[bump](x₀) = **0 exactly**, for any width and any amplitude.

The replicator **pre-registered this prediction before running any bump numerics** — then confirmed
it: the depression scales as **δ^1.0015 over 2.5 decades** of width, a 375× change end to end, with
**no trace of a plateau**.

Sub-findings:
- A weaker "offset" construction (probe at fixed *relative* position, scaling with δ) does approach a
  plateau — but drifts **~15% over 1.5 decades and ~40% over the full range**, outside its
  pre-registered 2% tolerance.
- **κ/π measured 1.07–1.10 vs the claimed 0.418 — off by a factor of ~2.6.**
- κ is exactly profile-independent (8 random profiles, identical at fixed x₀) but **not**
  x₀-independent: ~12% RMS, **1.64× across interior locations**. The claimed uniformity is not there.
- Proportionality of a_min to headroom: confirmed, but **near-trivial** — it follows from exact
  linearity of the DiHiTI operator (verified to ~1e−12).
- N-independence: **confirmed** to 0.0034%, at powers of two *and* non-powers of two. No common-mode
  power-of-two bug.

**Why this may be fatal.** V2a's pigeonhole argument needs Θ(N) disjoint supports carrying a
**uniform, N-independent amplitude bound**. If the depression scales linearly in δ, narrower bumps are
*weaker*, so a uniform bound over shrinking supports is exactly what fails.

**And it contradicts an earlier crew finding.** Cycle 1's Hand B asserted that "localized perturbations
get **cheaper** as they narrow, so Ω(N) is genuinely provable" — which is why H1 was revived. The
replicator finds narrower ⟹ **weaker** ⟹ more expensive. Both cannot be right.

**⚠️ The First Mate's own error is in scope here.** The dispatch prompt's statement of the claim was
*paraphrased by the First Mate* from red-team's summary, not quoted from `theory/v2_lower_bound.md`.
So there are two live possibilities and they must not be conflated:
  (a) V2a's proof is genuinely broken; or
  (b) the paraphrase misstated the construction and the replicator rigorously tested the wrong thing.

Either way the replicator's work stands — it found a real, sharp, pre-registered result. But **C30
cannot be promoted or dismissed until someone reads the actual proof.** Adjudicator dispatched.

### Cycle 3 — red-team returned: **C25 NARROWED. The `iff` is not an `iff`.**

Report: `results/cycle3_c25_hardening/redteam.md`. **C25 must NOT be promoted to `hardened` as worded.**

**The necessity half is false (C38).** Counterexample built on exact linearity of ε″ in g:
P = λ·(balanced corner pair at −0.40, +0.90) + (1−λ)·(balanced pair at ±0.20). Both pieces have g ≥ 0
and M₁ = 0; the support hull is **exactly** [−0.40, 0.90], so a+b = 1.30L — and crucially the
**effective support equals the nominal box**, closing the very ambiguity that produced the
replicator's earlier 76/80 confusion. At λ = 0.2: **min ε″ = +9e−9, passive.**

Confirmed through production `ht_derivative` at N = 2¹⁵/2¹⁷/2¹⁹, and independently by a hired hand
using erf smoothing, both grid conventions, non-power-of-two N, and its own closed form (λ* ≈ 0.278).

**Not a marginal exception.** Max passive corner-mass fraction: **26%** at a+b = 1.30L, 21% at 1.40L,
8.7% at 1.80L, 0.09% at 1.98L. **For every a+b up to 2L there exist passive profiles touching both
corners.**

**Consequence for the mission's headline.** The theorist's necessity line ("take the extreme pair")
establishes only a **sharpness statement about the box** — the same logical shape as Thm 4.2
("sufficient, constant L/2 sharp"). So C25 supplies **no necessary condition on any individual ε′**,
the billing as "the mission's first two-sided non-trivial result" was **overreach**, and **it does not
advance V6.** The First Mate propagated that billing to the Captain; corrected here.

**Five attacks failed — what survives is solid (C25′).** Sufficiency now looks genuinely proved: 3600
random many-atom balanced measures across 10 boxes, zero negatives; the Bauer/extreme-point step
checked against its *actual* hypotheses (F = min of linear functionals ⟹ concave u.s.c.; K weak*-compact
since a,b < L; Richter ⟹ ≤2 atoms). Cor 4.A re-derived by hand and exact. Boundary a+b = L is
**passive**, unambiguously, min ε″ = 0 at the interior point x* = b−a. Degenerate a = 0 and b → L both
survive. And the kernel/gauge (C1) was verified against production code to 4.5e−6 with a control
confirming the cos(πt/2L) factor is really present — **discharging the shared-assumption worry**, since
route 1 and (L1) do share the atom kernel and that assumption is now checked against ground truth.

### 🚨🚨 C39 — the finding with the widest blast radius

**The interior-window convention has its own blind spot.** With M₁ ≠ 0 the profile carries strict gain
matching C9 to 3–4 digits, **but the gain migrates to the seam** as M₁ shrinks: argmin at x/L =
−0.956, −0.9947, −0.99946, −0.99995 for M₁ = 1e−1 … 1e−4. **A 0.95L interior window — the replicator's
pre-registered choice, and the convention used in project scripts — reports PASS for every one.**

So *both* conventions fail, in opposite directions: whole-domain min is blind because the endpoint is
pinned to zero (C19); a fixed interior window is blind because near-balanced gain hides arbitrarily
close to the seam. **This affects corpus labels, not just C25** — and therefore potentially C33's
240/240, which may be agreement on a shared convention rather than a genuine fix. Relayed to the
auditor mid-flight with three specific checks.

**Free results also registered:** C40 threshold depth law (5-digit match), C41 (only the interval hull
matters; Cayley form is the *product* ξ₊|ξ₋| ≤ 1), C42 (the a+b = L negative minimum is quadrature,
O(h²) — a referee will ask).

### Cycle 3 — navigator returned: **THE MISSION IS NOT COMPLETE.** `NAVIGATION.md`, 439 lines.

It was dispatched to check whether the crew had been generous with itself, and told to contradict the
First Mate's synthesis. **It did.**

**It independently detected two adverse verdicts from disk artifacts alone** — C25's break and C30's
MISMATCH — while the ledger still carried them as good. (Both have since been integrated; it read a
snapshot from minutes earlier.) That it found them from `HEARTBEAT.log` and checkpoint files, with no
conversational context, is itself evidence the ledger-based handoff works as designed.

#### 🚨 The critique that matters most: **V3 is NOT met**

> (L1) is derived as an **identity**, so "passive ⟺ Φ_P ≤ Φ_ω" is ε″ ≥ 0 **in a change of variables** —
> logically the same move as Carathéodory–Toeplitz, **the very thing the Captain rejected** as "merely
> a re-representation of the HT in Fourier space."

The real content is **(L2)**, the extreme-point reduction — not (L1). And H9's actual deliverable, the
*minimal checkable characterization*, was never found. **The First Mate reported H9's unification to
the Captain in terms that implied V3; that was wrong, and the error is the same one the Captain
warned against in his very first substantive instruction.** Recorded as C43.

**V6 is only partially met.** 14 → 3 is real. But the Captain's critique (B) — *nothing is necessary
AND sufficient* — is unanswered again now that C25 has broken.

#### 🚨🚨 P0: the project has no certified numerical definition of PASSIVE (C44)

Whole-domain min is blind at the boundary (C19); a fixed interior mask is blind at the seam (C39).
That undecided convention sits underneath **the corpus labels, C18's α_c, C33's 240/240, and C36's
327/327**. Where ground truth and a candidate evaluator share a convention, their agreement is
**common-mode, not evidence** — the same failure that let the C22 bug pass an independent audit.

**Nobody owned this. Dispatched immediately at opus, since it gates every numerical claim.**

#### Objective audit of the victory conditions

| | Navigator's verdict |
|---|---|
| **V4, V5** | genuinely met (V4's *number*, not a closed form) |
| **V1** | substantially answered, but only **one** of its three levels is a theorem — and that one (C30) is contested |
| **V2a** | contested — "PROVED and tight" not supportable right now |
| **V2b** | met and correctly labelled, but **never adversarially tested** |
| **V3** | **NOT met** (C43) |
| **V6** | partially met |

Its read on C30: probably a **specification** failure — Lemma 1 takes the min over the bump
*neighbourhood*, while the ledger's compressed phrasing reads as evaluation at the bump *centre*,
where parity forces zero exactly. Consistent with the First Mate's paraphrase error. But it holds the
line: *a MISMATCH is a MISMATCH.*

#### Critical path (adopted)

**P0** seam adjudication (blocks all) → **P1** restate + re-replicate C30 → **P2** red-team C31, never
attacked → **P3** harden C36 → **P4** re-scope (L1) as a *frame* and gate (L2) as the theorem → **P5**
one referee pass. Further V1 speedup work **deprioritized**.

#### Still genuinely open — including one the crew had written off

- **N1 (new, best lead):** per-profile necessity may be about **mass concentration near the corners,
  not support** — from red-team's λ(a+b) corner-mass curve. The only live route to V6 as written.
- **N2:** T2/CFM — the highest-value untouched theory item, and precisely the gap C43 names.
- **N3:** **FFT-free O(N) is NOT excluded.** K14/K19/K21 died on quadrature and uncertified bounds,
  **not** on the absence of an FFT. The V1 negative's level 3 only bounds FFT-retaining approaches.

#### For the Captain

This is a **trigger-2 checkpoint** — board exhausted, direction needed — **not** trigger 1 (victory).
Publishable unit: **one paper, three parts** — I potential-theoretic structure, II the cost result
(*the novel half*), III Longhi's instability, **on which this voyage has done nothing**. Part III's
inclusion is the Captain's call because it decides **physics journal vs. math/CS**. Estimate: **two to
four more windows, not one.**

**And a note for `PARKED.md`:** the parked generation problem got materially easier *because of this
voyage* — (L2) + Bauer gives closed-form extreme points for the frontier optimization, replacing the
LP/SDP inner loop that made it expensive.

### Cycle 3 — auditor **CLEAN** (C33/C34/C35); adjudicator: **PARAPHRASE ERROR + PROOF NARROWED**

**Auditor: CLEAN on all three.** Verified the trigamma-DFT Clausen identity from scratch against
`mpmath` (machine precision, several M including non-powers of two); checked `gap_clausen`'s per-cell
integration against a fully independent **40-digit mpmath reference with no grid at all**, confirming
clean **O(h²)** convergence (ratio 4.00 over 7 doublings) versus DiHiTI's O(h). Reproduced 240/240,
then **5200 further agreements across 8 seeds × 5 power-of-two N × 5 non-power-of-two N — zero
disagreements**. C34's mechanism and C35's universal negative both reproduced, the latter by a fully
independent construction (exact δ-atoms, mpmath Clausen, exact closed-form balance) matching to 5–6
significant figures, with a deliberate counterexample hunt down to c = −0.01π finding none.

It also cleared the First Mate's mid-flight worry: the corpus uses **whole-domain min**, *not* the
fixed-interior-window convention C39 attacks, and refining all 240 profiles 8× and 16× produced **zero
label flips**. Flagged honestly as **empirical, corpus-specific clearance, not a general proof.**

**Ladder discipline:** C33/C34/C35 stay at `supported`. The auditor's mpmath work is replication-grade
in quality, but the ladder requires the *replicator* to sign the `replicated` rung, and consistency
matters more than convenience (C13 was held the same way).

#### C30 adjudication: **the error was the First Mate's, and it is confirmed**

Lemma 1 reads **`min_x [A(u₀+p)](x)`** — the minimum over the interior window. **The paraphrase
dropped "min_x".** The bump *is* symmetric about x₀, but the depression is *not* evaluated there. The
replicator rigorously tested a construction the proof never used.

**The replicator's fact was true and worth having.** σ = 0 really is a zero of the depression profile
(−ℋ[φ](0) = 0 to machine precision for *both* bump shapes). Lemma 1 uses the **minimum**, at
σ* = −0.723. And **all three** of its reported mismatches collapse to **one artifact**: it probed at the
bump *edge* rather than the extremum, where −ℋ is 0.388 vs 0.661 — ratio **1.70**, which reproduces the
κ factor 2.6, the 15% drift, and the 1.64× spread exactly. Measured properly κ spreads **1.078×**.

**Lemma 1 verified independently:** min_x Δ/a = −0.79758, flat to **0.1% over a 256× sweep in δ**, and
equal to the separately computed −max(−ℋ[φ]) to five digits.

**⭐ The cycle-1 contradiction dissolves (C45): all three measurements were right, of different
things.** Hand B measured L¹ mass (∝ δ — narrower *is* cheaper); the replicator measured Δ at the
centre (≡ 0 by parity); Lemma 1 measures L^∞ amplitude (flat). Nobody erred. And Hand B's
non-monotonic rise is a **window-edge artifact** — not the "under-resolved at 82 grid points"
explanation cycle 2 offered, **which was false**.

#### 🚨 The adjudicator's own narrowings matter more than the replicator's (C46, C47)

Attacking as instructed rather than merely refereeing, it found defects the replicator never reached:
constants self-contradictory (**the π is spurious — correct κ = 1/h_φ = 1.254**); `TV(p) ≤ 2r`
**vacuous** for bumps, so §1 contradicts its own §3.5; Theorem 2's intervals need insetting by 2kh.

**And the deep one (C46):** *any N-independent Lipschitz bound collapses Θ(N) to Θ(1)*, since a
flipping bump must eat the margin m₀ ≈ 1e−3·D. **Real ε′ profiles are smooth.** So V2a's lower bound
holds for a class admitting arbitrarily narrow features — which may **exclude the physically
realizable case entirely**. That partially revives **H2 and N3** (bandwidth; FFT-free O(N)), both
demoted on weaker grounds — and it means the mission's V1 "no" and V2 "yes" may both be answers about
a function class the Captain does not actually care about.

**Standing V2a result, honestly stated:** Θ(N), tight, for the finite (M+1)-input family in the
L^∞/BV ball of radius 1.68·D, **conditional on the class admitting O(1)-grid-point features.**
C31 (V2b) untouched — and still never adversarially tested (P2).

**Process fix already in force:** `PROTOCOL.md` now requires verbatim quoting of a claim from its
source when dispatching blind replication. This adjudication is what that rule was written to prevent.

---

## ⚓ ALL STOP — 2026-08-19. Captain aboard; voyage paused for discussion.

Both cron jobs **deleted** (hourly autonomous heartbeat `3b72d1d7`; 5-hour summary `cf43a4e9`).
Nothing will auto-resume. No crew are running — all cycle-3 hands returned, except P0 which was
terminated by the session limit **after checkpointing** ("E1 confirms my closed forms exactly.
Checkpointing before continuing"), so its partial work is on disk at
`results/cycle3_p0_seam_convention/`.

**To resume later:** re-create the hourly cron and re-dispatch from `BOARD.md` (P0 first — it still
gates every numerical claim). `RESUME.md` and this log are the handoff.
