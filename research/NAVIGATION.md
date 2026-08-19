# NAVIGATION — strategic assessment at the cycle-3 inflection

**Navigator, 2026-08-19.** Written after reading `MISSION.md`, `PROTOCOL.md`, `BOARD.md`,
`CLAIMS.md`, `KILLED.md`, `LOG.md` (all cycles), `PARKED.md`, `CAPTAIN_QUEUE.md`, `BUDGET.md`,
`MEASURE.md`, `RESUME.md`, `HEARTBEAT.log`, plus the primary reports in
`results/cycle3_c25_hardening/`, `results/cycle3_v2a_replication/`, `theory/v2_lower_bound.md`,
`theory/h9_potential_domination.md`.

This is a re-planning document. No experiments were run and no proofs attempted.

---

## 0. BOTTOM LINE

**The mission is not complete, and the gap between where it stands and where the synthesis in
`LOG.md` says it stands is larger than one rung.** Three things are true simultaneously:

1. **The voyage has produced real results.** The unification (L1)+(L2) is genuine intellectual
   work; eight challenger routes have died with documented mechanisms; the Amdahl decomposition
   (C36) is a clean, honest, publishable negative; α_c is settled; V5 is settled.
2. **Nothing is ESTABLISHED, and the two claims closest to it were both damaged by cycle-3 hands
   whose verdicts had not reached `LOG.md` when the synthesis was written.** The synthesis is
   therefore stale in exactly the direction the adversarial architecture exists to prevent.
3. **There is a cross-cutting numerical-integrity risk that no single hand owns**, discovered
   incidentally by red-team, which if real contaminates the corpus labels, C18, C33 and every
   PASS/FAIL verdict this voyage has issued.

The honest one-line status is: **V4 and V5 are done; V1 has a strong three-level negative answer
whose weakest level is the only one that is proved; V2a is contested; V3 and V6 are not met as
written.**

---

## 1. IS THE MISSION COMPLETE? — objective by objective, against the text as written

### ⚠️ First: two cycle-3 verdicts are on disk but not in `LOG.md` or `CLAIMS.md`

The First Mate's synthesis ("V1 is answered … the board's V1 items are exhausted") was written
after the numericist returned and **before** these two landed. Both are adverse. Both are in
`HEARTBEAT.log`. Neither is in the ledger.

**(a) C25 — the mission's advertised "first two-sided non-trivial claim" — has been broken as an
iff.** `results/cycle3_c25_hardening/CHECKPOINT.md` §2(c),(h). Red-team constructed an explicit
admissible drop measure with `a+b = 1.30L` whose support's convex hull *is* the nominal box, with
atoms sitting exactly on both corners, and it is **passive**: `min ε″ = +5.05e−9` reproduced
through production `ht_derivative` at N = 2^15, 2^17, 2^19. §2(h) then shows this is not a
near-threshold accident — for **every** box up to the maximum `a+b = 2L` there exist admissible
passive profiles touching both corners. Per-profile necessity fails everywhere.

What survives is the **box-quantified** statement: *for all admissible g supported in [−a,b],
ε″ ≥ 0 ⟺ a+b ≤ L* — i.e. `a+b ≤ L` is the maximal box on which passivity is automatic. Red-team
names the consequence itself: *"that is the same LOGICAL SHAPE as Thm 4.2 — so C25 is NOT the
mission's first two-sided per-profile result. It is a sharp sufficiency theorem with the region
generalised from symmetric to asymmetric."*

That verdict is correct and it is important. **C25 must be restated in `CLAIMS.md` and demoted
from `replicated` to `supported`** (the replicated statement is not the surviving statement), and
the LOG's claim that the mission has a two-sided non-trivial result must be withdrawn. Note the
replicator's own MATCH does not rescue it: the replicator verified sufficiency 104/104 and
resolved its necessity mismatches by *narrowing the construction*, which is precisely the regime
where red-team's counterexample lives.

**(b) C30 — the empirical core of the V2a proof — came back MISMATCH from blind replication.**
`results/cycle3_v2a_replication/replication.md`. The replicator reports that under the literal
reading (bump centred at x₀, depression measured at x₀) width-independence is **provably false**
by an exact parity argument (ℋ of an even function vanishes at its centre), with δ^1.00 scaling
confirmed over 2.5 decades; and that its own dilation-covariant construction gives a coefficient
κ/π ≈ 1.07–1.10 against the claimed 0.418.

**My reading, and I want the crew to test it rather than adopt it:** this is most likely a
**specification failure, not a refutation.** `theory/v2_lower_bound.md` Lemma 1 takes
`min_x [A(u₀+p)](x)`, i.e. the minimum of −a·ℋ[φ](σ) over σ, *not* the value at σ = 0. The
replicator was handed the ledger's compressed phrasing ("a bump … depresses ε″ by −a/(πC_φ)"),
which does read as evaluation at the bump. Its offset construction probes a fixed non-optimal σ,
which is exactly why it recovers 0.28–0.39 rather than 0.418 — an ordering consistent with
Lemma 1, not against it.

But **under `PROTOCOL.md` this does not matter for the rung.** A MISMATCH stands until resolved,
and the resolution is not the First Mate's to declare. The operational facts are:
- **C30 cannot be carried as "PROVED, tight."** It is `supported` with an open MISMATCH.
- The mismatch is partly the ledger's fault. The claim as written in `CLAIMS.md` is **wrong as
  literally stated**, and that is a real defect a referee would also find.
- The replicator produced two findings that stand on their own regardless: κ is **exactly**
  profile-independent across 8 independent base profiles (a clean, non-trivial confirmation nobody
  claimed), and N-independence holds to 0.0034% including three non-powers-of-two — the C22-class
  common-mode bug is genuinely absent here.

### Objective-by-objective

| Victory condition | Crew's implicit position | **Navigator's verdict** |
|---|---|---|
| **V1** — cheap certificate, or the demonstrated absence of one | "answered negatively at three levels" | **Substantially answered, but only one of the three levels is proof.** See below. |
| **V2a** — Ω(N) query bound | "PROVED and tight (C30/C31)" | **CONTESTED.** Blind replication returned MISMATCH on the load-bearing lemma. `supported`, not proved. |
| **V2b** — honest conditional barrier | "written as an explicit conditional barrier" | **MET at the level it claims**, and correctly labelled. Unreviewed by any adversary. `supported`. |
| **V3** — exact N&S, *"not a restatement of compute-the-transform-and-look-at-the-sign"* | "(L1) is exact and two-sided, 240/240" | **NOT MET as written.** See §1.1. |
| **V4** — closed-form prediction of the α crossover | "done, α_c = 0.66623" | **PARTIALLY MET, and the crew has quietly weakened it.** See §1.2. |
| **V5** — ε″(±L) = 0 | "settled, automatic" | **MET at `supported`, by two independent routes, auditor CLEAN.** The single cleanest item on the board. Still not `replicated` — but it is cheap. |
| **V6** — parsimony, one or two two-sided conditions, priors as corollaries | "14 → 3, nine corollaries" | **PARTIALLY MET, and weakened by the C25 break.** See §1.3. |

### 1.1 V3 is not met, and this is the crew's largest act of self-generosity

V3 demands N&S conditions *"stated as an explicit, checkable property of ε′ — **not** a restatement
of 'compute the transform and look at its sign'."* `MISSION.md` is emphatic about this: the Captain
already rejected Carathéodory–Toeplitz on exactly this ground, and the mission text says in terms
that *"a re-representation of the transform does not count as a win."*

(L1) is derived in `h9_potential_domination.md` §1 as **an identity**: `π ε″ = D(Φ_ω − Φ_P)`,
obtained by substituting the Cayley coordinate into the master identity. "Passive ⟺ Φ_P ≤ Φ_ω on ℝ"
is therefore *literally* "ε″ ≥ 0" written in a change of variables. It is exact and two-sided
because ε″ ≥ 0 is exact and two-sided. By the Captain's own stated test, **(L1) alone is the same
species of object as Thm 3.1 and CT — a re-representation, one coordinate change further along.**

That is not a dismissal of the work. The genuine content of cycle 1 is **(L2)**, which is *not* a
re-representation: the Bauer/Choquet reduction to extreme points is a real structural theorem that
demonstrably collapses the zoo and generated a sharper result. The mission's own H9 entry says the
target is *"the minimal checkable characterization of that domination"* — and **that is the part
that was not found.** §9 of the theorist's own report is labelled "partial"; K17 proved no
profile-independent finite node set exists; C26/(L3) gives a profile-dependent finite test that
runs at 0.2× speed. So the checkable characterization is exactly the open item.

**Recommendation:** stop describing (L1) as satisfying V3. Describe it as what it is — the correct
*frame*, from which (L2) and nine corollaries follow. V3 remains open, and its open form is sharp
and stateable: *find a checkable characterization of Φ_P ≤ Φ_ω that is not "evaluate Φ_ω − Φ_P."*

### 1.2 V4 is met numerically but not as written

V4 asks for **"a closed-form prediction"** of the crossover matching observation. What exists is
`α_c = 0.66623` from **bisection on the DiHiTI output** (C18) — an excellent, well-converged
*measurement*, independently audited, but a measurement. The closed-form route (H6, the Clausen dip
tolerance d*(b)) was explicitly **not confirmed**: C20 found the hypotheses are not met by this
family, and `clausen_general_form.py` was killed mid-run and never vectorized. The board still
lists H6 as "next up".

So: V4's *number* is in hand and is the mission seed's answer in the sense the Captain will care
about most. V4's *closed form* is not. Say both.

### 1.3 V6 is weaker than advertised, in two ways

- **The count is right but the shape is not.** 14 → 3 is real and is the best answer to critique
  (A). But V6 also demands the final characterization be **necessary and sufficient**. With C25's
  per-profile iff broken, the corpus is back to: one exact two-sided *identity* (L1, which is a
  re-representation), one structural reduction (L2), one profile-dependent finite test (L3), and a
  set of sharp **sufficient** conditions. Critique (B) — *"almost nothing is necessary AND
  sufficient"* — is **not yet answered.**
- **C35 closed the last named gap negatively.** Skew tolerance is zero, so unimodality does not
  extend. That is a good, honest result, and it also means the sufficient conditions cannot be
  widened toward necessity along that route.

### 1.4 V1 — the three levels are not of equal strength, and the crew states them as if they were

- **Level 1 (information, Θ(N)) is the only one that is a theorem** — and it is currently
  contested by its own blind replication. If the specification is repaired and Lemma 1 survives,
  this is the strongest thing the voyage owns.
- **Level 2 (method barrier, C31) is explicitly conditional** and correctly labelled. It is not
  evidence that no cheap test exists; it is evidence that *we cannot presently prove one doesn't*.
  It should never be listed alongside a proof without that qualifier.
- **Level 3 (Amdahl, C36) is a measurement of one implementation family on one machine.** It is a
  strong, well-executed engineering result and I believe it. But it bounds "approaches that retain
  an FFT", it is a wall-clock statement, and — importantly — **it bounds the wrong thing relative
  to the baseline it uses.** The FFT is 52–61% of the *incumbent's* cost; an FFT-free algorithm is
  not bound by this at all, and the mission's own scope note says so.

Net: V1's answer is **"no cheap test found, with eight documented failure mechanisms, one proved
information bound (contested), and a measured implementation ceiling."** That is genuinely
publishable and genuinely stronger than C15 was. It is not "answered negatively at three
independent levels" with the confidence the synthesis carries.

---

## 2. 🚩 THE CROSS-CUTTING RISK NOBODY OWNS — seam-localized gain vs. interior masks

This is the most consequential thing I found that is not on any board, and it was produced as a
side observation by red-team in `cycle3_c25_hardening/CHECKPOINT.md` §2(f):

> *M₁ = 0 is LOAD-BEARING and a KNIFE EDGE. Any M₁ ≠ 0 gives strict gain … **BUT the induced gain
> sits in a SHRINKING neighbourhood of the seam**: argmin = −0.956, −0.9947, −0.99946, −0.99995 as
> M₁ = 1e−1…1e−4. **A fixed interior window of 0.95L reports PASS for every one of these.** This is
> a real hazard for every numerical passivity verdict in the project, not just C25.*

Red-team is right, and the scope is wider than it says. The project is caught between two known
facts and has no validated instrument between them:

- **C19:** the min over the *whole* domain is blind — it reads exactly 0.000 for every passive
  profile, because the endpoints are structurally zero (C13/C21).
- **§2(f):** the min over a *fixed interior window* is blind in the opposite direction — genuine
  gain migrates into the excluded seam neighbourhood as the profile approaches balance.

Every PASS/FAIL verdict this project has issued lives in that gap. Specifically at risk:

| At risk | Why |
|---|---|
| **C18 (α_c = 0.66623)** | bisected on the **interior** minimum. If seam gain appears before the interior dip does, α_c is an **upper** bound, not the crossover. The auditor's 8× interior-window sweep tested *window size*, which is reassuring — but it was a sweep of interior windows, i.e. inside the blind zone. |
| **The 240-profile corpus labels** | produced by the C19 blind indicator; C28/T3 work used masks around 0.95–0.97L. |
| **C33's 240/240** | the headline agreement is between two evaluators **both** adjudicated under a mask. |
| **C25's counterexample itself** | `handA_results.json` shows minima at argmin ≈ ±0.9995L of order ±1e−8 with **sign varying across grid convention and N** (+4.5e−6, −1.3e−8, +1.6e−7, −4.3e−8). Red-team's own corroborating hand is reporting seam-noise at the same magnitude as the counterexample's +5.05e−9 margin. |
| **C36's 327/327 and 147/147 near-threshold** | correctness was measured against verdicts produced by the same adjudication convention. |

**This is a common-mode error of exactly the class `PROTOCOL.md` already warns about** — not a
shared library idiom this time, but a shared *adjudication convention*. It is not evidence that any
of these results is wrong. It is evidence that **the voyage has no certified answer to the question
"what does PASSIVE mean numerically?"**, and that question sits underneath every claim on the
ladder. A referee will ask it in the first paragraph.

**I rate this the single highest-priority item on the board**, above everything currently listed.
It is also cheap: the seam behaviour is analytically tractable (the Cayley map sends the seam to
η = ∞, and the near-seam asymptotics are exactly the Taylor expansion of (L1) that C8/C9/C13
already are — see `h9_potential_domination.md` §6).

---

## 3. THE CRITICAL PATH TO A DEFENSIBLE RESULT

Which claims must climb, and which are supporting detail:

**Must reach ESTABLISHED (these are the publishable unit):**
- **C30 + C31** — the V2 pair. This is the strongest and most novel thing the voyage owns.
- **C36** — the Amdahl ceiling. Cheap to harden, and it is what makes the negative result
  *practical* rather than merely asymptotic.
- **C23/C24 (L1 + L2)** — the unification. Needs to be *correctly scoped* before it is gated (see
  §1.1) or the referee rejects it on the Captain's own re-representation test.

**Need to be `hardened` but not necessarily `ESTABLISHED` (supporting detail):**
- C13/V5, C18/V4, C19, C21, C25 (in its surviving box-quantified form), C33.

**Do not need to climb at all:** C1–C12, C14, C16, C17, C20, C26, C27, C34, C35. These are
corollaries, mechanisms, or negative results that ride on the above. Spending adversary cycles on
them is the failure mode of a crew that has run out of hard problems.

### Ordered sequence — minimum work to a defensible result

**P0 — SEAM ADJUDICATION (blocks everything).**
*Adversary: `auditor`, then `red-team`.* Define and certify the project's passivity verdict.
Establish the near-seam asymptotic form of ε″ from (L1) (this is analysis, not numerics), derive
the correct excision rule, and **re-adjudicate C18, the 240 corpus labels, and C33 under it.**
Deliverable: one paragraph that a referee accepts as the operational definition of PASSIVE.
Nothing else should be gated before this lands, because every verdict downstream inherits it.

**P1 — REPAIR AND RE-REPLICATE C30.**
*Adversary: `replicator` (fresh hand, blind).* Rewrite the claim in `CLAIMS.md` to state the
minimum over the bump neighbourhood explicitly, with the bump shape, the definition of C_φ, and
the regime of validity in δ. Re-dispatch blind. **Do not resolve the MISMATCH by argument.** If
Lemma 1 survives a correctly-specified blind replication, V2a is the voyage's flagship; if it does
not, the Θ(N) bound falls and the synthesis's level 1 goes with it. Either outcome is worth more
than anything else on the board.

**P2 — RED-TEAM C31 (V2b).**
It has never faced an adversary. It is a *negative* claim about the reach of two proof techniques,
which is exactly the species this voyage has broken twice (K6/K7/K8 on Aug 13; C15 in cycle 1). It
must not go to a referee unattacked, and its named escape route R3 is the natural attack vector.

**P3 — HARDEN C36.**
*Adversary: `red-team`, then `auditor`.* Two specific attacks: (i) is `scipy.signal.hilbert` the
right FFT baseline, or does a real-input rfft/FFTW/pocketfft path change the 52–61% figure enough
to move the ceiling? (ii) the ceiling is machine- and N-specific — does it hold on a second
architecture? Cycle 1's `onehilb_rfft` result (~1.0×) is the load-bearing prior here and it was
produced by an agent that later withdrew a different measurement from the same session.

**P4 — RE-SCOPE AND GATE THE UNIFICATION (C23/C24).**
Rewrite (L1)'s claim text to say *frame*, not *criterion*, and put (L2) forward as the substantive
theorem. Then `replicator` on (L2) (the Bauer/Choquet step and the two extreme-point families are
independently checkable against Richter–Rogosinski and Khinchin), then `red-team`, then `referee`.
Red-team has already done half of this work for free in `cycle3_c25_hardening` §2(e).

**P5 — REFEREE, once.** A single referee pass over the assembled unit, not per-claim. Its charter
already permits rejection on sprawl; give it the whole thing and let it judge the shape.

**Explicitly deprioritized:** further V1 speedup attempts. Eight routes, three cycles, all dead,
with mechanisms. The marginal information from a ninth is near zero.

---

## 4. WHAT IS STILL GENUINELY OPEN

Ranked by research value, not by cheapness.

1. **The minimal checkable characterization of Φ_P ≤ Φ_ω (V3's real open form).** This is the
   mission's actual unfinished business and the crew has drifted past it. It is well-posed:
   K17 killed profile-independent node sets; (L3) gives a profile-dependent one; the gap between
   them is the question. Note this is *not* a speed question — it is a structure question, and V3
   never asked for speed.
2. **T2, the CFM/Choquet conjecture.** Would place (L1) inside a classical N&S framework and is the
   one route that could make the unification a *citable* theorem rather than a reframing. The
   theorist labelled it conjecture and never attempted it. `h9_potential_domination.md` §1 states
   the exact obstruction (the Riesz harmonic term) and the exact hoped-for mechanism (ω is Cauchy,
   all moments ≥1 infinite, affine part killed by M₁ = 0). **This is the highest-value untouched
   theory item on the board.**
3. **An FFT-free O(N) algorithm.** Level 3 of the V1 answer bounds only FFT-retaining approaches,
   and the mission text admits it. The honest position is that this is *not excluded*, and every
   attempt has failed for reasons that were about *pruning bounds and quadrature* (K14, K19, K21),
   not about the absence of an FFT. A fast-multipole evaluation of the log potential with a
   *certified* per-node bound is still, in principle, the right shape — K14 died on constants with
   an uncertified bound, which is a different death.
4. **R3 — bounded-coefficient linear decision trees.** Correctly identified as the one crew-sized
   escape route on V2b. It is the difference between "we cannot prove it" and "here is what would
   have to be true."
5. **Where does per-profile necessity actually live?** C25's break is not just a demotion — it is a
   new question, and a good one. §2(h) shows passive profiles touching both corners exist for
   **every** box up to 2L, with a computed maximal corner-mass fraction λ(a+b) (0.262 at 1.30L
   down to 0.0009 at 1.98L). That λ curve is a real, unexplained object. **The per-profile
   necessary condition the mission has been hunting since day one may be a statement about mass
   distribution near the corners rather than about support.** Nobody has looked.
6. **Quietly dropped items I want back on the board:** H6's closed form (V4's actual wording; the
   general Clausen reconstruction was killed mid-run and never revived); H12 (the reopened
   approximate-certificate class from K16 — reopened in cycle 1 and never worked); the corpus
   re-labelling by interior minimum (RESUME item 5, superseded by P0 but subsumed into it); and
   H10 (the 2024 sparse Nichtnegativstellensätze paper, still unread — it is aimed squarely at
   open item 1 above).

---

## 5. WHAT THE CAPTAIN SHOULD BE TOLD

**This is a checkpoint** — but under trigger 2 (the board is exhausted of *V1* hypotheses and needs
direction), **not** trigger 1. No claim has reached ESTABLISHED and none is close. Telling him the
mission is substantially complete would be wrong.

What to say, in this order:

1. **His bonus question has an answer, and the answer is no.** There is no test cheaper than
   computing the DiHiTI. Eight named routes died with documented mechanisms; the practical ceiling
   is measured below 3× because the FFT is 52–61% of the work and is irreducible; and there is a
   query-complexity argument that no algorithm can read fewer than Ω(N) samples. **State plainly
   that the last of these is currently contested by its own blind replication** and that the crew
   believes the contest is a specification defect it introduced, but has not yet proved that.
2. **His original question — the α crossover — is answered numerically: α_c = 0.66623**, converged,
   independently audited, and the reason he could not see the pattern is now understood (C19: his
   diagnostic reads exactly 0.000 for every passive profile, so it had no resolution to give). This
   is the finding he will most immediately recognize as his.
3. **The parsimony critique got a real answer: 14 statements → 3**, with nine former theorems now
   corollaries, and his "two complementary theorems" tell resolved — they are one theorem applied
   to two families. **But** the corpus still has no per-profile necessary-and-sufficient condition:
   the one candidate (Cor 4.B) was broken by red-team this cycle, and (L1) is exact but is a
   re-representation of the transform by his own stated test. **Critique (B) is not yet answered.**
4. **The publishable unit, and its shape.** I see one paper, and it is a *negative-result plus
   structure* paper, roughly:
   > *"Passivity of spatial-Kramers–Kronig media: a potential-theoretic characterization, and the
   > cost of deciding it."*
   > **Part I (structure).** DiHiTI as a logarithmic potential; passivity as potential domination;
   > the extreme-point reduction; the existing sufficient conditions recovered as corollaries;
   > sharp asymmetric-support law. *This is the part with the citations and the referee appeal.*
   > **Part II (cost).** Deciding passivity requires Ω(N) queries; the transform is optimal up to a
   > log factor; the log factor is blocked by a method-relative barrier stated as conditional; and
   > the constant-factor route is capped below 3× by Amdahl. *This is the part that is novel in a
   > way nobody has published.*
   > **Part III (physics).** Longhi's unfollowed remark: gain in an sKK medium can seed a growing
   > interface mode that destroys reflectionlessness. This is why the decision problem matters and
   > it is the paper's motivation, currently underweighted in the voyage.
   Part II is the more novel half; Part I is the more citable half. They need each other.
5. **What it would take to get there.** The five-step critical path in §3, of which P0 (defining
   what PASSIVE means numerically) and P1 (repairing and re-replicating the Ω(N) lemma) are
   prerequisites for everything. Realistically this is **two to four more full-throttle windows**,
   not one.
6. **One thing only he can decide:** whether Part III (the Longhi instability) is in scope. It is
   in `MISSION.md` as the motivation, it is the reason a physics journal takes the paper rather
   than a numerical-analysis one, and **the voyage has done no work on it at all** (K5 killed the
   eigenvalue-screening route in Wave 2 and nothing replaced it). If he wants it, it is a mission
   item and needs a hand. If he does not, the paper is a math/CS paper and should be aimed there.

---

## 6. IS THERE A BETTER VOYAGE?

**Not yet — but the parked problem is now closer to the right next mission than it was, and for a
reason the crew did not anticipate.**

Assessed honestly:

- **Finishing this mission is still the right immediate move.** The critical path in §3 is short,
  concrete, and ends in a defensible publication. Abandoning it at `supported` would waste three
  cycles of genuinely good adversarial work and leave the Captain with nothing citable. **Do not
  switch voyages now.**
- **The parked flipped problem (generation) has gained an argument in its favour that `PARKED.md`
  does not contain.** The decision problem has now been answered as far as it goes *and the answer
  is negative at every level*: no cheap test, Θ(N) queries, no per-profile N&S condition, skew
  tolerance zero. A field where every question resolves negatively is a field that is telling you
  the interesting object is somewhere else. Meanwhile the generation side already has a complete
  parametrization (Thm 6.1), a constructive band-limited form (Cor 6.2), a working constructor, and
  a budget–performance frontier showing a two-step profile matching the logistic's six-order
  suppression at **half** the absorption budget. That last number is a *design* result with direct
  bearing on the Captain's actual paper (JOPT-113210) and his actual coatings.
- **However — and this is the navigator's real recommendation — I do not think the next voyage is
  either of these.** The strongest unexploited asset this voyage produced is **(L2), the
  extreme-point reduction**, and its natural home is *neither* deciding nor generating: it is
  **optimization over the passive cone**. (L2) says certifying a convex family reduces to
  certifying its extreme points, and Bauer's principle says the *extremum* of a concave functional
  over that family is attained there too. That is precisely the machinery the frontier problem
  (Wave 3 Open Problem 8.2) needs, and it converts an LP/SDP-inner-loop numerical study into a
  problem with closed-form extreme points. **The parked voyage is more tractable now than when it
  was parked, because this voyage built its tool.** That is worth telling the Captain, and it is
  worth writing into `PARKED.md` regardless of when he acts on it.
- **One genuinely new candidate voyage, from §4 item 5:** the λ(a+b) corner-mass curve red-team
  produced while breaking C25. If per-profile passivity is governed by mass concentration near the
  corners rather than by support, that is a *different characterization theorem* than anything in
  the corpus, it is two-sided by construction, and it would answer critique (B) — the one part of
  the Captain's structural critique this voyage has not touched. It is speculative. It is also the
  only lead on the board that could still deliver V6 as written.

---

## 7. SPECIFIC DISAGREEMENTS WITH THE FIRST MATE'S SYNTHESIS

Recorded plainly, as instructed.

1. **"V1 is answered … at three independent levels."** The three levels are not independent in
   strength: one is a contested theorem, one is explicitly conditional, one is a machine
   measurement of one implementation family. The synthesis presents them as three pillars; they are
   a pillar, a scaffold, and a floorboard. The *conclusion* is probably right. The *confidence* is
   not earned.
2. **"the board's V1 items are exhausted."** True for speedups. Not true for V1 as a research
   question: the FFT-free route is unexcluded by the crew's own scope statement, and H12 (reopened
   in cycle 1 when K16 died) was never worked at all.
3. **"C25 … the mission's FIRST TWO-SIDED non-trivial claim … leading candidate for the voyage's
   first established claim."** Broken this cycle, by the crew's own red-team, with a counterexample
   verified on production code at three grids. `CLAIMS.md` still carries it at `REPLICATED` with a
   ⭐⭐. That is the single most misleading line in the ledger right now.
4. **"V2a PROVED — tight."** Its blind replication returned MISMATCH. Under `PROTOCOL.md` rule 3
   this is a demotion trigger, not a footnote.
5. **"(L1) … is exact and two-sided"** offered as satisfying V3. Exact and two-sided, yes — because
   it is an identity. V3 explicitly excludes restatements of the transform, and the Captain has
   already rejected one object of this species. This needs re-scoping before a referee sees it.
6. **A process observation.** The synthesis was written from three returned hands while a fourth
   (red-team, on the flagship claim) was still working, and the two hands that returned after it
   both returned adverse verdicts. The gate held — the hands did their jobs and wrote to disk. What
   failed is **integration**: `LOG.md` and `CLAIMS.md` are three verdicts behind reality. Suggest a
   standing rule: **no synthesis while an adversary is in flight on a claim it names.**

---

## 8. IMMEDIATE ACTIONS (in order, before any new science)

1. Integrate the two cycle-3 verdicts into `LOG.md`, `CLAIMS.md`, `KILLED.md`. Demote C25 to
   `supported` in its surviving box-quantified form; mark C30 `supported / MISMATCH open`.
2. Open the seam-adjudication item (P0) at the top of `BOARD.md`.
3. Restate C30 in `CLAIMS.md` precisely enough to be blind-replicated (min over neighbourhood;
   bump shape; C_φ definition; δ regime), then re-dispatch `replicator`.
4. Re-scope C23 from "criterion" to "frame" in `CLAIMS.md`; promote (L2) to the substantive claim.
5. Add to `PARKED.md`: (L2) is the tool the frontier problem needs — the parked voyage got easier.
6. Ping the Captain per §5. This is a trigger-2 checkpoint, and he should be told plainly that the
   mission is **not** complete.
