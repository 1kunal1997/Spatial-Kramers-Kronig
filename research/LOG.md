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
