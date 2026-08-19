# RED TEAM — C25 ("Cor 4.B"), cycle 3 hardening attempt

**VERDICT: NARROWED** (the necessity half, as written, is FALSE; the sufficiency half is
*strengthened* — I found the proof is airtight and I sharpened its statement)

Claim under attack, exactly as registered in `CLAIMS.md` and in
`research/theory/h9_potential_domination.md` §4:

> **Cor 4.B.** `g ≥ 0`, `M₁ = ∫ g tan(πt/2L) dt = 0`, `supp g ⊆ [−a, b]`.
> Then `ε″ ≥ 0` **iff** `a + b ≤ L`.

All work in this directory. Two independent evaluators throughout: the closed-form atom kernel
(`common.exact_kernel`) and the **production** `tmm_helper.ht_derivative`.

---

## Headline

**The `iff` is not an `iff`.** Read as a statement about a profile — which is how it is written,
how it is registered, and how it was billed to the Captain ("the mission's FIRST two-sided
non-trivial result") — it is refuted by explicit counterexamples, for **every** box, not merely
near threshold. The theorist's own proof line, *"Necessity: take the extreme pair at t₊ = b,
t₋ = a,"* establishes something strictly weaker: that the box `a + b ≤ L` is the largest box on
which passivity is **automatic**. That is a *sharpness* statement about the hypothesis region,
and it is the **same logical shape as C3/Thm 4.2**, which is already registered as
*"sufficient, constant L/2 sharp."*

So C25 does not change the logical type of what the mission knows. It generalises the region
from symmetric to asymmetric intervals — a real and useful advance, and the sufficiency proof is
sound — but **it supplies no necessary condition on any individual ε′**, and therefore does not
advance V6 (which asks for two-sided conditions on ε′) and is not the mission's first two-sided
per-profile result.

---

## Attack 1 — the `iff`, per-profile. **LANDED. Claim refuted as written.**
`e1_counterexample.py`, `e11_degenerate.py`

Because `ε″` is exactly linear in `g` (C21, 6e−15), admissible measures may be superposed. Take

```
P = λ·(balanced corner pair at −0.40, +0.90)  +  (1−λ)·(balanced pair at −0.20, +0.20)
```

Each piece is `≥ 0` and has `M₁ = 0`, so `P` is admissible. `supp P = {−0.40, −0.20, +0.20, +0.90}`,
whose convex hull is **exactly** `[−0.40, +0.90]`: `a + b = 1.30 L > L`, and the *effective*
support equals the *nominal* box — atoms sit **on** both corners. This is precisely the
nominal-vs-effective distinction the replicator flagged, closed off.

| λ | min ε″ (closed form, \|x\| ≤ 0.9999L, atoms excised) | verdict |
|---|---|---|
| 1.00 | −1.004e−1 | FAIL |
| 0.50 | −1.940e−2 | FAIL |
| **0.20** | **+9.0e−9** | **PASSIVE — counterexample** |
| 0.10 | +7.0e−9 | PASSIVE |
| 0.01 | +5.2e−9 | PASSIVE |

Confirmed through **production `ht_derivative`** on the smoothed staircase at N = 2¹⁵ / 2¹⁷ / 2¹⁹:
`min ε″ = +5.05e−9, +5.05e−9, +4.79e−9` — all PASS, stable over a 16× range in N. Not a
closed-form artifact and not near-threshold: **26% of the mass sits on the two corners of a 1.30 L
box** and it is still passive.

**And this is not an edge effect — it holds for every box.** Bisecting the largest corner-mass
fraction λ that remains passive (remaining mass in a balanced pair at ±0.1 L):

| box a+b | max passive corner fraction λ |
|---|---|
| 1.30 L | 0.262 |
| 1.40 L | 0.205 |
| 1.80 L | 0.087 |
| 1.94 L | 0.0018 |
| 1.98 L | 0.0009 |

For **every** `a + b` up to the maximum possible `2L` there exist admissible `g` supported on
`[−a, b]`, touching both corners, that are passive. Per-profile necessity fails everywhere on the
parameter range, not in a boundary buffer.

### Independent re-verification of the counterexample (hand A)

Because this counterexample is the whole verdict, I dispatched a hand to rebuild it with a
deliberately different construction: **erf/Gaussian-CDF** smoothed steps (not raised cosine),
smoothing width swept 0.0025–0.02, **both** grid conventions (endpoint-excluded *and*
`np.linspace`), N = 1024/4096/16384 plus the **non-powers-of-two** N = 6250 and 10000, exclusion
radius swept 0.02–0.15, plus its own hand-derived closed form as a second evaluator.
Report: `handA_report.md`. It confirms:

- λ = 0.2 and λ = 0.1: **passive.** The only negative values are a vanishing edge artifact at the
  last retained node (argmin → ±0.9999 as N grows, magnitude O(1e−7) and shrinking with the
  smoothing width: −2.6e−6 → −1.3e−7 → −7.9e−9 → exactly 0 as s = .02 → .0025). Pulling the edge
  cut in: min over |x| ≤ 0.99 is **+7.1e−5**, ≤ 0.98 **+2.9e−4**, ≤ 0.95 **+1.8e−3** — positive and
  growing inward.
- λ = 0.5 and 1.0: genuinely negative, N-independent to 3–4 digits at fixed interior x
  (−1.946e−2 at x = 0.721; −1.006e−1 at x = 0.500) — matching my closed form to 3 digits.
- **The two grid conventions do not disagree** (< 2e−7 absolute at λ = 0.1, 0.2; < 6e−4 relative at
  λ = 0.5, 1.0; same argmins; no sign flips). The C22 `linspace` bug is confirmed absent.
- Its own λ-scan puts the sign change at **λ\* ≈ 0.278** (with the middle pair at ±0.20); my
  bisection gave 0.262 with the middle pair at ±0.10. Consistent.

Its verdict, unprompted and in its own words: *"this measure has support [−0.40, +0.90], so
a + b = 1.30 > L = 1, yet λ = 0.2 with M1 = 0 is passive — on this family 'a + b ≤ L' is **not
necessary** for passivity."*

### The narrowed claim the evidence *does* support

> **C25′ (sharp balanced-support law — corrected quantifiers).**
> Let `g ≥ 0` on `[−L, L]` with `M₁ = 0`, and let `a = −inf supp g`, `b = sup supp g`.
> 1. **(sufficiency, per profile — PROVED, see Attack 2)** `a + b ≤ L ⟹ ε″ ≥ 0`. The nominal box
>    may be used in place of the effective support; the statement is then only conservative.
> 2. **(sharpness, about the BOX — not about the profile)** For every `a + b > L` there *exists*
>    an admissible `g` supported in `[−a, b]` with `ε″ < 0`, namely the balanced corner pair.
>    Equivalently: `∀g (g ≥ 0, M₁ = 0, supp g ⊆ [−a,b]) : ε″ ≥ 0  ⟺  a + b ≤ L`.
>    The `iff` binds the **box**, and the quantifier sits **outside** it.
> 3. **(the converse is false)** For every `a + b ∈ (L, 2L)` there also exist passive admissible
>    `g` whose support touches both corners. `a + b > L` implies nothing about a given profile.
> 4. **(boundary)** `a + b = L` is **PASSIVE**, marginally: `min ε″ = 0`, attained at the interior
>    point `x* = b − a`.

**Free strengthening (no extra work).** The same `≤2`-atom reduction shows the interval is not a
restriction: for an *arbitrary* support set `S`, the family `{g ≥ 0, M₁ = 0, supp g ⊆ S}` is
uniformly passive **iff** `a + b ≤ L` with `a = sup{−t : t ∈ S, t < 0}`, `b = sup{t : t ∈ S, t > 0}`.
Only the interval hull of `S` matters — non-interval supports buy nothing.

---

## Attack 2 — break the sufficiency half. **FAILED. It is solid, and I now believe it is proved.**
`e6_sufficiency_stress.py`

- 3600 random balanced many-atom measures (1, 2, 3, 5, 10, 25 atoms) across 10 boxes with
  `a + b ≤ L`, including the extreme `[−0.02, 0.98]` and `[−0.9, 0.1]`: **zero negatives**, worst
  `min ε″ = +9.1e−11` (grid noise).
- Corner-loaded extreme points at `a + b = L`: `min ε″ = 0` at exactly the predicted `x* = b − a`.
- The proof structure is legitimate, and I checked each step rather than taking it:
  - `F(g) = min_x ε″(x; g)` is a minimum of linear functionals ⟹ **concave and u.s.c.**;
  - `K = {P ≥ 0 on [−a,b], mass 1, ∫tan dP = 0}` is weak\*-**compact** (needs `[−a,b]` compact and
    `tan` continuous on it, i.e. `a, b < L`, which `a + b ≤ L` supplies except in the degenerate
    `a = 0` case, handled separately below);
  - **Bauer's minimum principle** ⟹ the min of a concave u.s.c. function is attained at an
    extreme point; Richter/Rogosinski (2 moment constraints) ⟹ extreme points have **≤ 2 atoms**;
  - `t₋ ≤ a`, `t₊ ≤ b` ⟹ `t₋ + t₊ ≤ a + b ≤ L` ⟹ **Cor 4.A**.

## Attack 3 — break Cor 4.A, the load-bearing lemma. **FAILED. I re-derived it and it is exact.**

I redid the algebra independently rather than trusting the write-up:

```
Δ′(η) numerator = (β+γ)[ η(1+βγ) + (γ−β) ]        — degree 1 ⟹ exactly ONE critical point
η* = (β−γ)/(1+βγ) = tan(A−B),  and η* ∈ (−γ, β) where (η−β)(η+γ)(1+η²) < 0 ⟹ η* is a MAX
outside (−γ, β):  Δ rises from −∞ to 0⁻  (M₁ = 0 ⟹ Δ ≈ −M₂/2η² < 0)
⟹ sup_ℝ Δ = max(Δ(η*), 0)
Δ(η*) = γ ln(sin B/cos A) + β ln(sin A/cos B)      — I verified (★) symbolically
A+B ≤ π/2 ⟺ sin A ≤ cos B and sin B ≤ cos A ⟺ both logs ≤ 0
```

Airtight, **given the kernel** (Attack 5). Cleaner restatement worth recording: in the Cayley
coordinate `ξ = tan(πt/2L)`, Cor 4.A is **`ξ₊·|ξ₋| ≤ 1`** — a *product*, not a sum. `a + b ≤ L` is
its shadow on interval boxes.

**Free new result (`e12_extra.py`) — the threshold depth law.** Just past threshold, the extremal
unit-mass corner pair has

```
min ε″  =  −(D/2)·sin(π a/L)·(a + b − L)/L  +  O((a+b−L)²)
```

Numerics vs the analytic slope, 5 significant figures, symmetric in `a ↔ b`:

| (a, b) | measured slope | analytic slope | ½·sin(πa/L) |
|---|---|---|---|
| (.50,.50) | −0.50000 | −0.50000 | 0.50000 |
| (.25,.75) | −0.35355 | −0.35355 | 0.35355 |
| (.10,.90) | −0.15450 | −0.15451 | 0.15451 |
| (.05,.95) | −0.07821 | −0.07822 | 0.07822 |

The violation is **linear**, not quadratic — so the threshold is sharply resolvable, which is a
point *in favour* of the corrected claim's practical usefulness.

---

## Attack 4 — are all four hypotheses load-bearing? **All four are. One is a hidden knife edge.**
`e5_hypotheses.py`

| hypothesis | load-bearing? | evidence |
|---|---|---|
| `g ≥ 0` | **yes, decisively** | 287/400 random **signed** measures with `M₁ = 0` supported in the *middle half* `[−L/4, L/4]` are non-passive; worst `min ε″ = −1.795`. Without monotone `ε′` the box condition gives nothing. |
| DiHiTI gauge | **yes** | deleting the `cos(πt/2L)` seam normalisation shifts `ε″` by `−∫g ln cos > 0`, breaks `ε″(±L) = 0` (`+0.1103`), and the `a+b ≤ L` threshold ceases to mean anything. The threshold is a statement about **this gauge only**. |
| `supp g ⊆ [−a,b]` | yes (Attack 1 fixes *which* statement) | — |
| `M₁ = 0` | **yes — and it is a KNIFE EDGE with a numerical trap** | see below |

### ⚠️ The `M₁ = 0` knife edge, and a hazard that reaches beyond C25

Any `M₁ ≠ 0` gives strict gain (`M₁ = 0` is necessary, C8), with depth matching C9's
`−M₁²/(2πM₂)` to 3–4 digits — an independent corroboration of C9. But **the induced gain migrates
to the seam as `M₁ → 0`**, in a box (`a + b = 0.5 L`) that is deep inside the safe region:

| M₁ | min ε″ over \|x\| ≤ 0.99999 L | argmin | C9 prediction | min ε″ over **\|x\| ≤ 0.95 L** |
|---|---|---|---|---|
| 1e−1 | −1.0970e−3 | −0.95609 | −1.094e−3 | −1.0759e−3 |
| 1e−2 | −1.3265e−5 | −0.99470 | −1.326e−5 | **+9.29e−4 (PASS)** |
| 1e−3 | −1.3552e−7 | −0.99946 | −1.355e−7 | **+1.13e−3 (PASS)** |
| 1e−4 | −1.3578e−9 | −0.99995 | −1.358e−9 | **+1.15e−3 (PASS)** |

A fixed interior window of `0.95 L` — which is exactly what the replicator's *pre-registered*
protocol specified, and what several project scripts use — **reports PASS for every one of these
gain-carrying profiles**. This is a hazard for *any* numerical passivity verdict in this project,
not only for C25: gain from imperfect balance is real, is quantitatively predicted, and is
invisible to a windowed minimum. It should be recorded on the board.

## Attack 5 — common-mode error: is the kernel itself (C1) right, and is the replication genuinely independent of (L1)? **The concern is real; I discharged it.**
`e2_gauge.py`

The replicator's route 1 "re-derived the periodic Hilbert transform of a point mass from scratch"
— but that object **is** the master identity C1 restricted to atoms, which is exactly the
foundation of (L1). Route 1 and (L1) therefore share their single load-bearing assumption. The
claimed independence is independence *of derivation*, not *of assumption*. So I tested the shared
assumption against the only thing that outranks it — the production code that **defines** DiHiTI:

| t₀ | N = 2¹⁴ | N = 2¹⁶ | N = 40000 |
|---|---|---|---|
| 0.00 | 7.20e−5 | 4.50e−6 | 1.21e−5 |
| 0.30 | 5.22e−5 | 7.58e−6 | 1.24e−5 |
| −0.60 | 9.00e−5 | 2.05e−5 | 3.35e−5 |
| 0.85 | 2.38e−4 | 6.14e−5 | 9.99e−5 |

(max abs error of `ht_derivative` vs `(1/π)ln[cos(πt₀/2L)/|sin(π(x−t₀)/2L)|]`, converging in N.)
**Control**: comparing against the same kernel *without* the `cos` gauge factor is off by exactly
`ln cos(πt₀/2L)` (3.67e−2 at `t₀ = 0.3`; 4.63e−1 at `t₀ = 0.85`) — so the gauge factor is really
there and is not fitted.

**Explicit guard against the C22 / `np.linspace` failure mode** (`PROTOCOL.md`): I used an
endpoint-**excluded** periodic grid `x_j = −L + 2Lj/N` throughout, and ran both power-of-two and
`N = 40000`. The theorist and replicator used `linspace`-style grids. The two conventions agree,
so the C22-class bug is not in play here. *(An independent hand was additionally dispatched to
re-derive the Attack-1 counterexample with erf/tanh smoothing on both grid conventions — see
`handA_report.md`.)*

## Attack 6 — the boundary case `a + b = L` exactly. **Unambiguous in exact arithmetic; a practical caveat in floating point.**
`e6_sufficiency_stress.py` §C, `e10_boundary_smooth.py`

Exact answer: **PASSIVE**, marginally. `Δ(η*) = 0` identically at `A + B = π/2`, so `min ε″ = 0`
attained at the interior point `x* = b − a`; verified at `(a,b) = (.5,.5), (.25,.75), (.1,.9),
(.02,.98), (.001,.999)`, matching the predicted `x*` to 5 decimals.

**Caveat.** Pushed through production `ht_derivative`, the corner-atom measure at `a + b = L`
returns a small **negative** minimum at every finite N: `−1.5e−9, −9.1e−11, −5.7e−12` at
N = 2¹⁵/2¹⁷/2¹⁹, i.e. `O(h²) → 0`. I isolated the cause: at fixed `N = 2¹⁸` the value is
`−2.286e−11` **independent of the smoothing width** over `w = 10h … 320h` (a 32× range), so it is
`gradient`/`cumtrapz` **quadrature**, not the smoothed step straddling the corner and violating
the support hypothesis (which was my first hypothesis, and it was wrong). For genuinely smooth `g`
strictly inside the box at `a + b = L` the interior margin is a healthy **+1.1e−3** — matching
C21's `~1e−3·D`. So: the exact claim is correct at the boundary, and the numerical verdict there
converges to zero **from below** and is unreliable at any finite N. A referee will ask; state it.

## Attack 7 — degenerate corners. **All survive.**
`e11_degenerate.py`

- `a = 0`: with `g ≥ 0` on `[0, b]` and `tan ≥ 0` there, `M₁ = 0` forces `g = c·δ₀`, whose
  `ε″ = −ln|sin(πx/2L)|/π ≥ 0` — trivially passive for **any** `b ≤ L`. The degenerate case where
  weak\*-compactness of `K` needs `b < L` is therefore vacuous, not a gap.
- `b → L`, `a = L − b → 0`: PASS down to `a = 1e−8`, despite the Cayley weight ratio reaching
  `2.5e−16`. No conditioning failure.
- multimodal / many-atom / random: covered in Attack 2 (3600 measures, zero negatives).

---

## What I could not break, stated plainly

The **sufficiency** half and its sharpness are, in my judgement, correct and now better supported
than before this shift: I re-derived Cor 4.A by hand, verified `(★)` symbolically, checked the
Bauer/extreme-point step against its actual hypotheses, stressed it with 3600 adversarial
measures, verified the kernel it rests on against production code on a deliberately different grid
convention, and closed the degenerate corners. I found no parameter corner where it fails.

## Residual risks that would still worry me

1. **The `M₁ = 0` windowing hazard (Attack 4).** It is not a defect in C25, but it means every
   *numerical* passivity verdict in this project — including the corpus labels and the 3600 checks
   above — is only as good as its window. Settling experiment: re-run the 240-profile corpus's
   verdicts with the window at `0.99999 L` and with `M₁` measured, and see how many labels move.
2. **Everything rests on the kernel C1**, which is still registered at `conjecture`. I verified it
   to `4.5e−6` and converging, on atoms only. It should be promoted or attacked on its own; C25 and
   (L1) both fall with it. Settling experiment: extend the `e2_gauge.py` comparison to continuous
   `g` and to `L ≠ 1`, and get C1 onto the ladder.
3. **`Δ` on the two outer branches.** I argued `Δ < 0` on `(β, ∞)` and `(−∞, −γ)` from the
   `M₁ = 0` asymptotic `Δ ≈ −M₂/2η²`. That is sound for a 2-atom measure with finite `M₂`, which
   is all Cor 4.A needs, but it is an asymptotic argument I checked numerically rather than a
   rigorous bound on the whole branch. A referee could ask for the monotonicity argument in full.

## Recommended ledger actions

- **C25 → NARROWED**, restated as **C25′** above; it should **not** be promoted to `hardened` in
  its current wording. Once reworded, it has passed a specific and thorough attack on the
  surviving half and I would sign SURVIVED for **C25′ parts 1, 2 and 4**.
- **Strike the billing.** "The mission's first two-sided non-trivial claim" is overreach: the
  corrected statement has the same logical shape as C3, and supplies no necessary condition on any
  individual `ε′`. It does **not** advance V6.
- **New claims worth registering** (all first stated here): the effective-support/arbitrary-`S`
  strengthening; the Cayley form `ξ₊|ξ₋| ≤ 1`; the threshold depth law
  `min ε″ ≈ −(D/2)sin(πa/L)(a+b−L)/L`; the seam-migration hazard for `M₁ ≠ 0`.

## Files

`common.py` (shared evaluators, endpoint-excluded grid) · `e1_counterexample.py` ·
`e2_gauge.py` · `e5_hypotheses.py` · `e6_sufficiency_stress.py` · `e10_boundary_smooth.py` ·
`e11_degenerate.py` · `e12_extra.py` · `CHECKPOINT.md` (resumable state) ·
`handA_report.md`, `handA_staircase.py`, `handA_lamscan.py`, `handA_results.json`
(independent re-verification of Attack 1 — hand A, erf smoothing, both grid conventions)
