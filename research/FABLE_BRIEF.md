# BRIEF FOR A SOLO SESSION — the one open question

You are working **alone**. Do not spawn subagents; there is no crew and no budget for one.
Read `research/STATE.md` first — it is short and current. Do **not** read `research/LOG.md`
end-to-end; it is long history and will waste your context.

---

## THE ONE QUESTION

For a real permittivity profile ε′(x) on the window [−L, L], the DiHiTI construction
(differentiate → periodic Hilbert transform → integrate → de-tilt) returns a partner ε″(x).

**Find conditions on ε′ that are NECESSARY AND SUFFICIENT for ε″ ≥ 0 everywhere.**

Constraints that make an answer acceptable:
- **Two-sided.** A sufficient-only or necessary-only condition is not an answer. Every previous
  attempt has produced one-sided conditions; that is the failure mode to avoid.
- **Parsimonious.** One or two conditions, with existing special cases falling out as corollaries.
  Newton's laws, not a thirteenth theorem beside twelve others. Adding another one-sided sufficient
  condition to the pile is a *failure* even if it is true and novel.
- **A criterion, not a re-representation.** "ε″ ≥ 0 rewritten in different coordinates" does not
  count — that objection has already sunk two prior attempts (Carathéodory–Toeplitz, and arguably the
  potential-domination form). The test is whether your condition can be checked *without* effectively
  computing ε″.

**Bonus (secondary):** conditions checkable faster than computing the transform and taking the
minimum. Do not spend your time here — eight routes have died. Prioritise the two-sided condition.

---

## WHAT IS ALREADY DONE — do not re-derive

Three statements carry everything (details in `research/theory/h9_potential_domination.md`):

1. **The potential frame.** The construction is a logarithmic potential of the drop density
   g = −dε′/dx. Passivity ⟺ a comparison of two potentials. *Treat this as a frame that makes proofs
   possible, NOT as an answer — it is derived as an identity.*
2. **The extreme-point reduction** (Bauer/Choquet). A convex family is entirely passive iff its
   **extreme points** are. This is where the real content lives, and it is the tool that made
   everything below provable.
3. **Node collapse.** An n-atom drop gives ≤ n candidate test points — an exact grid-free test.

Best current **sufficient** condition: g ≥ 0, balance ∫g·tan(πt/2L)dt = 0, support in [−a, b] with
**a + b ≤ L**. (The old "middle half" rule is just the symmetric case a = b = L/2.)

Also settled: ε″(±L) = 0 automatically; skew tolerance is exactly zero; a corrected evaluator agrees
with all 240 test profiles.

---

## ⭐ THE LIVE LEAD — start here

**a + b ≤ L is sufficient but NOT necessary.** Passive profiles exist with a + b all the way up to 2L.
The counterexample: a convex mixture of two balanced atom pairs, support hull exactly [−0.40, 0.90],
passive at mixing fraction λ = 0.2.

The pattern in the counterexamples — the maximum passive **corner-mass fraction** as the box widens:

| a + b | max passive corner mass |
|---|---|
| 1.30 L | 26% |
| 1.40 L | 21% |
| 1.80 L | 8.7% |
| 1.98 L | 0.09% |

**So necessity appears to be about HOW MUCH MASS SITS NEAR THE CORNERS, not about where the support
ends.** A condition of the form "mass near the endpoints, weighted somehow, must not exceed a
threshold" is the most promising route to a genuine two-sided statement. Nobody has pursued it.

Second lead, untouched: whether the potential comparison is an instance of the **Cartier–Fell–Meyer /
Choquet dilation order**, which would supply classical necessary-and-sufficient machinery. The
obstruction: the test cone here is only logarithmic potentials, not all subharmonic functions.
(Cartan's domination principle and balayage are *false friends* — both are one-way.)

---

## TRAPS — every one of these has already cost this project real time

1. **"Passive" has no certified numerical definition.** Whole-domain min ε″ is blind: it reads
   exactly 0.000 for *every* passive profile, because ε″(±L) = 0 is an operator identity. A fixed
   interior window (e.g. 0.95 L) is *also* blind: when balance is violated the gain migrates to the
   seam (argmin → 0.99995 L as M₁ → 1e−4). **Decide and state your convention, and check it is not
   blind in either direction.**
2. **`np.linspace(-L, L, N)` duplicates the endpoint** under periodic continuation. This produced a
   bogus 1.70× speedup that survived an independent audit because both implementations used the same
   idiom. Use a deliberate grid convention; test at non-power-of-two N.
3. **`tan(π/2)` overflows to 1.6e16** at the window edge, injecting an O(h) offset proportional to the
   edge slopes. This masqueraded as a defect in a theorem for a full cycle.
4. **Parity.** The Hilbert transform maps functions even about a point to functions odd about that
   point — so a symmetric bump produces exactly zero depression *at its own centre*. Probing there
   measures nothing.

---

## VERIFICATION STANDARD

**Before reporting any number, assert a control that would FAIL if your harness were wrong** —
disable the component and check the number moves, or compare against a closed form, or test a case
whose answer you know independently. State the control. *A number without a control is a conjecture.*

Label everything **proved / verified numerically / conjectured**. A derivation with a gap is a
conjecture; name the gap precisely. This work will be attacked afterwards by an adversarial reviewer,
a blind replicator, and a numerical auditor — every prior session's confident assertions were caught
by them, so labelling honestly now saves rework later.

**Checkpoint to disk every ~15 minutes** (goal / established / ruled out / in progress / next step),
so nothing is lost if the session ends. Write to `research/theory/` and `research/results/`.
**Write only inside `research/`** — the rest of the repo is read-only.

## Useful files

- `research/STATE.md` — current truth, short. Start here.
- `research/MISSION.md` — full objectives and history
- `research/BOARD.md` — open hypotheses (N1 is the corner-mass lead)
- `research/theory/h9_potential_domination.md` — the three statements
- `research/results/cycle3_c25_hardening/redteam.md` — the counterexamples and the corner-mass curve
- `theory/passivity_2026Aug/` — prior waves (read-only)
- `tmm_helper.py` — `ht_derivative` is the production pipeline (read-only)
