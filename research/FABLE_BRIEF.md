# BRIEF FOR A SOLO SESSION — 2026-08-19 (second handoff)

You are working **alone**, outside the crew. Do not spawn subagents. Read `research/STATE.md`
first — the banner at the top is current and supersedes older text in that file. Do **not** read
`research/LOG.md` end-to-end; it is long history and will waste your context.

**If you are the same lineage as the session that wrote `research/theory/ns_answer.md`: that was
excellent work and it has now cleared two independent gates. Read §2 below anyway — three things
in it did not survive contact with the adversaries, and one of them is a harness bug you should
know about before you write another line of numerics.**

---

## 1. WHAT CHANGED SINCE THE LAST BRIEF

The last brief asked for necessary-and-sufficient conditions on ε′ for ε″ ≥ 0, or a rigorous proof
that none exist. That session answered **both halves** and the Captain has read it. Current state:

**Gated and standing:**
- **C49 (Theorem 2 — no finite battery decides passivity) is `replicated`.** A blind replicator,
  given only the 20 atoms and two weight vectors with the originating code forbidden, reproduced
  it by an independent route (direct x-space log-potential evaluation, no Cayley coordinates, no
  FFT). Three independent routes now agree on the headline number: −7.756e−3 / −0.007756 /
  −7.755719e−3 at x = +0.63111.
- **The F5 audit returned CLEAN** on C48–C54 (`research/results/f5_audit_fable/audit.md`), with the
  exact node test independently reconfirmed to 9–10 digits by mpmath.
- **N1 closed, N2/CFM refuted, Lemma B proved.** All standing.

**The Captain's amendment (this is the important part — `MISSION.md`, "Captain's Amendment"):**
The end goal is now two sequential goals.

- **GOAL A (V7):** since N&S conditions appear not to exist, deliver a **mathematically rigorous
  and exact proof of why they cannot**. "Proved modulo a numerically-certified ingredient" does not
  satisfy V7. **The crew is finishing this right now — see §4. It is not your task.**
- **GOAL B (V8):** **your task.** See §3.
- **V6 amended:** the two-sidedness requirement is withdrawn (C49 makes it unachievable). The
  **parsimony** requirement is retained in full and now binds Goal B.

---

## 2. THREE THINGS FROM THE LAST SESSION THAT DID NOT SURVIVE — read before writing numerics

None of these changes a theorem's truth value. All three are worth your attention anyway.

**(a) 🔴 The harness had the `linspace` endpoint bug.** `common.py`'s `production_min_eps2` defaults
to `grid="linspace"`, and `np.linspace(-L, L, N)` duplicates the endpoint, so the FFT-based Hilbert
transform sees period `2L + dx` instead of `2L`. Verified exactly, twice: `N·dx − 2L` is **exactly
`+dx`** for linspace and **exactly `0`** for the periodic grid (N = 2048, 2049, 4096).

This is what C53's "profile-dependent O(h) offset, mechanism unidentified" actually was. It is also
**the single most expensive error class in this voyage's history** — `PROTOCOL.md` records it
producing a bogus 1.70× speedup that survived an independent audit because both implementers reached
for the same idiom. It reappeared undetected inside your own harness.

Consequence: at **N = 2¹⁹** — one power of two below the range quoted, and a value used elsewhere in
that same session — the default grid reports **PASS for the k=16 witness, which genuinely has GAIN**.
`grid="periodic"` is already implemented in the same function, is correct at N = 2¹⁴, and is ~3
orders of magnitude more accurate at matched N. **Pass `grid="periodic"` in everything you run.**
The "cancels in differences" half of C53 is robust under both conventions and stands.

**(b) The k=16 witness is not quotable; the k=8 witness is.** k=16 took 30 boundary-walk tries,
`|P_ker grad| = 1e−4`, kernel quality 2.7e−11, singular values down to 8.2e−08. The k=8 witness is
robust and independently confirmed three ways. **Quote k=8.** Also note the k=8 battery is **rank 7,
not 8** on the chosen pairs — harmless (the proof needs only dim ker ≥ 2) but a referee will ask.

**(c) `e1_density.py`'s √s shrinkage is stated in η-space, where it is non-monotone** for η₀ = 8
(widths 10.3 → 43.5 → 7.8 → 3.4 → 1.86). In x-space it is monotone (0.290 → 0.084 → 0.054 → 0.031
→ 0.018), so it is a Cayley-stretch artifact and Theorem 1 is unaffected — but the document quotes
only the η₀ = 1.7 row, and an adversary will find the η₀ = 8 column.

---

## 3. ⭐ THE ONE QUESTION FOR THIS SESSION — Goal B (V8)

> **Find the cleanest, most succinct set of conditions on ε′ that are sufficient-but-not-necessary
> and/or necessary-but-not-sufficient for passivity.**

**Read the word "cleanest" as binding. This is a demand for FEWER statements, not more.** The corpus
already holds roughly a dozen one-sided results. Producing a thirteenth is a *failure*, exactly as it
was under the old brief — what changed is only that the target is now allowed to be one-sided. The
deliverable is the **minimal spanning set**:

**(B1) The weakest sufficient condition.** One condition on g = −dε′/dx implying passivity, from
which these fall out as **corollaries**:
- **C3 / Thm 4.2 (middle-half, sharpened to C25′):** g ≥ 0, M₁ = 0, supp g ⊆ [−a, b] with a + b ≤ L.
- **C5 / Thm 5.2 (unimodality):** g even and unimodal ⟹ passive at **any** width, no support limit.
- **C6 / C7:** the plateau Clausen closed form and the dip tolerance d*(b).

Wave 3 states C3 and C5 are *"genuinely complementary: neither contains the other"* — two sufficient
conditions, neither implying the other. The Captain's diagnosis is that this is the signature of an
incomplete theory. Find their common parent.

**(B2) The strongest necessary condition.** One condition from which the **C8 cascade** (edge-slope
match, M₁ = 0, M₂ ≥ 0, σ ≥ 0) falls out as corollaries. These four look like the leading terms of a
single expansion — the numericist derived an edge expansion under P0,
ε″(L−d) = (1/π)[M₁·e + (M₂/2)e² + …], which is the obvious place to start. If they *are* one
statement, the collapse is mechanical.

**(B3) The gap.** Quantify what sits between B1 and B2 — profiles neither certifies. An honest,
quantified gap is part of the deliverable, not an admission.

### What counts as success

Mechanical test: **each prior one-sided theorem must be derivable from your statement in one or two
lines, and you must actually perform those derivations.** The count of independent one-sided
statements in the corpus must strictly decrease. A new condition sitting *beside* C3 and C5 rather
than *above* them fails V8 by construction.

**A rigorous proof that no parent exists also satisfies V8** — it bounds the minimal spanning set
from below, which is a real result. Do not treat that outcome as a consolation prize. But prove it;
do not conclude it from having failed to find one.

### One dead end already checked — do not re-walk it

The obvious first guess is *"g lies in the convex cone generated by balanced pairs each with
a + b ≤ L."* That cone is genuinely passive (a positive sum of nonpositive Δ's is nonpositive) and is
weaker than C3. **But it does not contain C5:** an even unimodal g decomposes into symmetric pairs
(−t, +t) with a + b = 2t, which exceeds L as soon as t > L/2, yet C5 says such g is passive at any
width. The parent, if it exists, is strictly weaker than that cone.

Untested directions, ranked by my guess at promise: (i) a majorization / rearrangement condition —
the domination *is* an ordering, and orderings usually have a rearrangement characterization;
(ii) a condition stated on the log-potential directly rather than on g's shape or support; (iii) a
decomposition into pieces where balance is restored only in aggregate, which is what would let a wide
unimodal g through; (iv) balayage — but note the full subharmonic/CFM order was **refuted** (C52), so
only the log-potential cone is available.

---

## 4. 🚧 WHAT IS NOT YOURS — the crew owns these, do not touch them

Duplicated effort is the one thing that would make this handoff net-negative.

- **Lemma E / the thickening step / Theorem 2's analytic closure (board F1, Goal A).** A crew
  theorist is finishing it now and has a complete analytic route: a symmetric reciprocal-pair base
  point p₀ (atoms ξ = ±2, ±½, weight ½ each) with Δ_{p₀}(η) = ½ln[|η²−4||η²−¼|/(1+η²)²], touching
  zero **only** at η = 0, with Δ″(0) = −25/4 exactly and an IFT continuation with non-vanishing
  Jacobian determinant −(25/4)ln2. I verified all of it independently. It lands in
  `research/theory/lemmaE_closure.md`. **Leave it alone.**
- **Red-teaming C49.** A crew adversary gets this once F1 lands. You wrote the theorem; you are the
  wrong seat to attack it.
- **P0 (the seam convention) and the C53 fix.** Crew items.

If you finish B1/B2/B3 early, **stop and write up** rather than expanding scope.

---

## 5. TRAPS — every one has already cost this project real time

1. **Pass `grid="periodic"`.** See §2(a). This is the live one.
2. **"Passive" has no certified numerical definition.** Whole-domain min ε″ is blind: it reads
   exactly 0.000 for *every* passive profile, because ε″(±L) = 0 is an operator identity. A fixed
   interior window is *also* blind: when balance is violated the gain migrates to the seam. **State
   your convention and check it is not blind in either direction.** Prefer the exact grid-free node
   test (`exact_sup_gap` in `results/fable_ns_answer/common.py`) — it is audited CLEAN.
3. **`tan(π/2)` overflows to 1.6e16** at the window edge, injecting an O(h) offset ∝ the edge slopes.
4. **Parity.** The Hilbert transform maps functions even about a point to functions odd about that
   point — a symmetric bump produces exactly zero depression *at its own centre*. Probing there
   measures nothing.
5. Test at **non-power-of-two N**. The linspace bug hid for two cycles because N was a power of two.

---

## 6. VERIFICATION AND WRITE-UP STANDARD

**Before reporting any number, assert a control that would FAIL if your harness were wrong** —
disable a component and check the number moves, compare against a closed form, or test a case whose
answer you know independently. **State the control.** *A number without a control is a conjecture.*

Label everything **proved / verified numerically / conjectured**. A derivation with a gap is a
conjecture; name the gap precisely. Your last session's honest labelling is the reason its results
survived the gates — keep doing exactly that.

**Checkpoint to disk every ~15 minutes.** The Captain's window closed mid-cycle today and killed
three agents. Two had checkpointed and lost nothing; **one had written nothing to disk and lost ~16
minutes of opus work entirely.** Write goal / established / ruled out / in progress / next step /
blockers, and keep it accurate in place. On this task the **ruled out** section is the most valuable
thing you can leave behind — the eliminations *are* the progress, and a successor cannot reconstruct
them.

**Write only inside `research/`.** Document to `research/theory/minimal_oneside.md`; scripts and
outputs to `research/results/fable_goalB/` with a `manifest.json`. Call
`research/hb.sh fable-goalB "<note>"` at every checkpoint.

---

## 7. USEFUL FILES

- `research/STATE.md` — current truth, short. **Start here** (top banner first).
- `research/MISSION.md` — "Captain's Amendment", V7/V8, and the structural critique that defines
  what "clean" means to the Captain.
- `research/CLAIMS.md` — the claim register and rungs; the C49 scope note and the C53 refinement are
  at the bottom.
- `research/theory/h9_potential_domination.md` — **the frame you will work in**: (L1) one law,
  (L2) extreme-point reduction, (L3) node collapse.
- `research/theory/ns_answer.md` — your last session's document.
- `research/results/fable_ns_answer/common.py` — audited evaluators (`exact_sup_gap` is the one to
  use; remember `grid="periodic"` for the production wrap).
- `research/results/f5_audit_fable/audit.md` — what the auditor checked and how.
- `theory/passivity_2026Aug/profile_generator.py` — labelled sample profiles (read-only).
- `theory/figures/2026Apr27/`, `theory/figures/2026May07/` — the labelled example corpus; dozens of
  profile families with PASS/FAIL labels. Go here if you are short of test cases.
