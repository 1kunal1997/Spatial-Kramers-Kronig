# Adjudication of C30 / V2a — cycle-2 red team vs cycle-3 blind replicator

*Adjudicator (red team, cycle 3), 2026-08-19. Scripts: `adj1_law.py`, `adj2_position.py`,
`adj3_narrow.py`, `adj4_reconcile.py`, all in this directory, all rerunnable.*

# VERDICT: **PARAPHRASE ERROR** (primary) — and, on independent grounds, **PROOF NARROWED**

The replicator did careful, correct work on a construction **the proof never uses**. Its central
mathematical fact (ℋ[even](centre) = 0) is true, is reproduced here to machine precision, and is
**perfectly consistent with the proof** — it says the probe point σ = 0 is a *zero* of the
depression profile, whereas Lemma 1 evaluates the depression at its *minimum* over σ. The engine of
the proof is verified here, independently and quantitatively, to be **exactly right**.

Separately, and not because of the replicator, the document `theory/v2_lower_bound.md` carries three
real defects (a self-contradictory constant, a false excuse for an old anomaly, and one overreaching
sentence in §1) and needs one technical patch. C30 should be restored — **narrowed**, not as
originally worded.

---

## 1. What the proof actually claims and constructs (task item 1)

Quoting `theory/v2_lower_bound.md`:

> **Lemma 1.** … Then, for δ small …,
>   **min_x** [A(u₀ + p)](x) = ε″₀(x₀) − a/(π C_φ) + O(δ)

and, in the proof body:

> For x = x₀ + δσ with σ = O(1), the leading part gives Δ(x₀ + δσ) = … = −a·H[φ](σ) + O(aδ) …
> Hence **min_σ Δ = −a·max_σ(−H[φ](σ))**

The measured quantity is therefore **the minimum over x of the perturbed ε″ across the whole
interior window** — *not* the value at x₀. This is not a reading; it is literally what the code does.
Both cycle-2 scripts bisect on

```python
def imin(v, x): return v[np.abs(x) <= FRAC*L].min()      # FRAC = 0.95
f = lambda a: imin(dihiti(u0 + sgn*a*cbump(x, x0, half), x), x)
```

(`v2a_compact.py`, identically in `v2a_hard_family.py`). The bump **is** symmetric about x₀; the
depression is **not** measured at x₀. The proof's probe point is at a *fixed relative position*
σ* inside the bump — precisely the replicator's "offset" construction — with the crucial extra
condition that σ* is the **argmin**, i.e. **−H[φ]′(σ*) = 0**.

## 2. The replicator's fact is true, and harmless (task item 3)

I checked it specifically rather than assuming. `adj4_reconcile.py`, on a standalone fine grid:

| shape | max(−ℋ[φ]) | at σ = | −ℋ[φ](0) |
|---|---|---|---|
| C^∞ compact bump (cycle-2's shape) | 0.79758 | −0.7231 | −0.0000 (exact) |
| raised cosine (replicator's shape) | 0.66107 | −0.5728 | −0.0000 (exact) |

ℋ[φ](x₀) = 0 exactly, both shapes. **Confirmed.** It kills nothing, because σ = 0 is a zero of a
function whose *minimum* is what Lemma 1 uses.

Sanity check that this is the whole story: in `adj1_law.py` at N = 2¹⁶ I measured Δ(x₀)/a for the
cycle-2 bump and it goes as **1/δ**, not δ — because with `np.linspace(-L,L,N)` and even N, x = 0 is
not a grid point, so the "value at x₀" is really the value half a grid step off centre,
≈ −a·ℋ[φ]′(0)·(h/2δ). Both the replicator's δ¹ (its own grid/base profile) and my δ⁻¹ are readings
of a quantity that is identically zero in the continuum. Nothing physical lives at σ = 0.

## 3. Lemma 1 is correct — verified independently and quantitatively

`adj1_law.py`, N = 2¹⁶, compact C^∞ bump at x₀ = 0, **physical** half-width δ swept 256×:

| grid pts per half-width k | δ/L | **min_x Δ / a** |
|---|---|---|
| 16 | 4.88e−4 | −0.78853 |
| 32 | 9.77e−4 | −0.79456 |
| 64 | 1.95e−3 | −0.79700 |
| 128 | 3.91e−3 | −0.79750 |
| 256 | 7.81e−3 | −0.79758 |
| 512 | 1.56e−2 | −0.79758 |
| 1024 | 3.13e−2 | −0.79742 |
| 2048 | 6.25e−2 | −0.79675 |
| 4096 | 1.25e−1 | −0.79407 |

Flat to **0.1 %** over a 256× range in physical width, and equal to the *independently computed*
−max(−ℋ[φ]) = −0.79758 to five digits. Lemma 1's δ-independence is **true and exact**, and the
mechanism (Hilbert kernel homogeneous of degree −1) is the correct one.

Confirmed a second way with a different bump family and a different observable
(`adj3_narrow.py` (ii)): Gaussian bump at **fixed** x₀ = 0, flip amplitude a_min/D = 1.8724, 1.8697,
1.8694, 1.8694, 1.8701, 1.8734 for σ/L = 0.05 … 0.001 — **flat to 0.2 % over 50× in width**, down to
16 grid points per σ.

## 4. Why the replicator got 1.07–1.10 instead of 0.418, and a 15 % drift (task item 2b)

Both are one artifact: **it probed at the edge of the bump, not at the minimum.** With the bump
centred at x₀ + δ/2 and the probe at x₀, the probe sits at relative coordinate σ ≈ −1 (or −1/2,
depending on whether δ is its full width or half-width). From the table above, for its raised cosine:

- −ℋ[φ] at the minimum: **0.661**; at σ = −1: **0.388**; ratio **1.70**. At σ = −1 for the C^∞ bump
  the ratio is **1.64**.
- Its measured depression/a ≈ 0.28–0.30 ⇒ its κ = 1/0.30 ≈ 3.3, κ/π ≈ **1.06** — exactly what it
  reported. The correct value 1/0.661 → κ/π = **0.482** (raised cosine), 1/0.7976 → **0.399** (C^∞).
  The **factor ≈ 2.6 is the ratio of an off-extremum value of −ℋ[φ] to its maximum.**
- Its **~15 % drift** has the same origin: at σ* the derivative of −ℋ[φ] vanishes, so the value is
  second-order insensitive to any O(δ) or grid perturbation (hence my 0.1 % flatness). At σ = −1 the
  derivative is large, so the same O(δ) corrections show up at first order. Its own numbers
  (−0.281 → −0.393) are the O(δ) term it was standing directly on top of.
- Its **x₀-spread of 1.64×** is the same number again — its κ ran 0.852 → 1.401, ratio **1.64**,
  which is exactly max/edge for a compact bump. Measured properly (`adj2_position.py`, 13 interior
  positions, N = 2¹⁵): min_x Δ/a ranges −0.7822 … −0.8118 (**3.6 %**) and κ = a_min/ε″₀(x₀) ranges
  1.2146 … 1.3089 (**1.078×, 1.8 % RMS**), κ/π = 0.387–0.417. **x₀-independence holds**, to the O(δ)
  Lemma 1 advertises. The replicator's 12 %/1.64× is not a property of the transform.

Grid-convention control (`adj2_position.py`): `endpoint=True` (cycle-2's convention, the C22 bug) vs
`endpoint=False` change every entry in the sixth significant figure. **The C22 endpoint bug is not
load-bearing here.** The replicator's N-independence result (0.0034 %) is confirmed and matches
cycle 2's.

## 5. Cycle-2's headline number reproduced from first principles

`v2a_compact.py`'s uniform bound r/D = **1.680** at k = 4 was never explained in the document. It is
this: at k = 4 (bump on 8 grid points) the discrete operator gives min_x Δ/a = **−0.67910**
(`adj1_law.py`), and max ε″₀/D = 1.1411, so r/D = 1.1411/0.6791 = **1.680**. Exact match. The k-drift
1.680 → 1.500 → 1.447 the document waved at as "the O(δ) correction" is in fact
1/0.679 → 1/0.761 → 1/0.789, i.e. **the approach of the discrete bump to the continuum shape** — a
resolution effect, converging *upward* in k toward 1.1411/0.79758 = 1.431 (which `adj2_position.py`
measures directly at k = 64: 1.4318). The document read the direction of that convergence backwards.
This does not damage the theorem — 1.680 is a valid finite N-independent bound — but the stated
explanation is wrong.

Small-k behaviour (`adj3_narrow.py` (i)): min_x Δ/a = −0.380 (2 grid points), −0.526 (4), −0.679 (8),
−0.761 (16). **Bounded and nonzero even at 2 grid points** — so the construction does not collapse at
the finest packing, and M = Θ(N) is safe for any fixed k ≥ 1.

## 6. Resolving the cycle-1 contradiction (task item 4)

Hand B (M4) concluded "localized perturbations get **cheaper** as they narrow". The replicator
concluded "narrower ⇒ **weaker** ⇒ more expensive". **Both are wrong as stated, and neither
contradicts Lemma 1**, because all three measure different things:

| who | observable | scaling | correct? |
|---|---|---|---|
| Hand B | L¹ mass ‖p‖₁ of the flipping bump | ∝ δ → 0 | true, but L¹ is not the query-model norm |
| Replicator | Δ at the bump's own centre | ≡ 0 in the continuum | true, but that point is a zero, not the minimum |
| Lemma 1 | min_x Δ (= L^∞ amplitude cost) | **flat in δ** | ✅ verified here to 0.1 % over 256× |

Hand B's *stated* conclusion also rests on a second error the document then excused incorrectly. M4's
a_min falls to σ/L = 0.02 and then **rises** (1.87e−3 → 2.49e−3 → 2.90e−3). The document explained
this away as "a resolution artifact — σ/L = 0.02 is ≈ 82 grid points, and below it the bump is
under-resolved". **That excuse is false**: 82 grid points is amply resolved, and §3 above shows the
law is flat down to 16. I reproduced M4 (`adj3_narrow.py` (iii)) and found the real cause: the
easiest position is **always x₀* = +0.97 L**, i.e. *outside* the 0.95 L interior window. The
depression lobe sits at |x − x₀| ≈ 0.72 δ; once δ shrinks below the 0.02 L gap to the window edge the
lobe falls outside the scored region entirely, so only leakage counts and a_min must rise. It is a
**window-edge artifact, not a width law**. Hand B's headline was driven by the easiest position
migrating to the seam (where ε″₀ is small), not by narrowness.

So: **Lemma 1 is right; Hand B's conclusion was right for the wrong reason and its non-monotonicity
was misdiagnosed by the cycle-2 document; the replicator's finding is right about a quantity that is
identically zero.**

## 7. What still has to change — the narrowing (task items 2c, 5)

These are *my* attacks on the proof, independent of the replicator. None of them is fatal; all of
them must be fixed before this is quoted.

**(N1) The constant is self-contradictory and the π is spurious.** §1 says "κ = π C_φ⁻¹ ≈ 1.31";
Lemma 1 says "κ = π C_φ"; §3.2 says "C_φ = 1/0.418 ≈ 2.39". These three cannot all hold
(π·2.39 = 7.5 ≠ 1.31). The correct, checkable statement has no π in it at all:

> Δ_min = −a·h_φ with h_φ := max_σ(−ℋ[φ](σ)) = **0.79758** for the cycle-2 C^∞ bump;
> a_min(x₀) = ε″₀(x₀)/h_φ, i.e. **κ = 1.2538**, measured κ = 1.215–1.309 (§4 above).

The table column "a_min/(π ε″₀)" = 0.418 is κ/π; its analytic value is 0.399, and the residual 4.7 %
is the k = 8 resolution offset, not physics. The document should report h_φ, not C_φ.

**(N2) §1's "not an artifact of admitting … arbitrarily rough inputs" is an overreach, and it
contradicts the document's own §3.5.** The perturbations are bumps of *fixed height* on a *fixed
number of grid points*, so as N → ∞ their derivative diverges like N. The ball 𝒰(u₀, r) is bounded in
L^∞ and in TV, but **TV is vacuous here**: TV of a bump of height a is 2a *whatever its width*, so
"TV(p) ≤ 2r" adds exactly nothing to "‖p‖_∞ ≤ r" for this family. The class is not smoothness-
constrained in any sense that excludes grid-scale spikes.

Sharper, and stronger than the band-limit caveat the document does state: **any N-independent
Lipschitz bound ‖p′‖_∞ ≤ C collapses the theorem.** A flipping bump needs height
a ≥ m₀/h_φ (it must at least eat the interior margin m₀ ≈ 1e−3·D, C21), hence width δ ≳ m₀/(C h_φ),
hence at most M ≤ 0.95L·C·h_φ/m₀ disjoint supports — a constant **independent of N**. So the
construction yields Ω(min(N, LC h_φ/m₀)), and the Θ(N) is delivered entirely by admitting features at
the grid scale. That is the honest regime statement, and it should sit in §1, not only in §3.5.

**(N3) A technical patch to Theorem 2.** The proof scans positions out to |x₀| = 0.95L, the same
window over which the minimum is scored. Positions within ~δ of that edge lose their depression lobe
(this is exactly the M4 pathology of §6). Fix in one line: place the M intervals inside
[−0.95L + 2kh, 0.95L − 2kh]. M is still Θ(N) and no constant changes materially. Without the patch
the "41/41 positions flip" claim is fragile at the two extreme positions.

**(N4) x₀-independence is O(δ) + a seam term, not exact.** min_x Δ/a drifts monotonically
−0.7822 → −0.8118 from x₀ = −0.9L to +0.9L (3.6 %) — the regular tan(πt/2L) part of ∂_tK, exactly as
Lemma 1's proof predicts. The document's "not on δ, on x₀, or on N" should read "up to O(δ) and
O(dist to seam)". Harmless: the theorem needs only the *sup*.

## 8. What stands (task item 5)

**Standing, and now independently verified:**

> **Lemma 1 (verified).** For a compact C^∞ bump φ of unit height and any width δ small compared with
> the scale on which ε″₀ varies, min_x A(u₀ + aφ((x−x₀)/δ)) = ε″₀(x₀) − a·h_φ + O(δ) with
> h_φ = max_σ(−ℋ[φ](σ)) = 0.79758, independent of δ, of N, and of x₀ to O(δ). Verified to 0.1 % over
> 256× in δ, 0.2 % over 50× for a Gaussian, 1.8 % RMS over 13 interior positions, and unchanged under
> both grid conventions.

> **Theorem 2 (stands, with patch N3).** Fix k ≥ 1. There is a base profile u₀ and a radius
> r = 1.68·D, independent of N, such that any deterministic adaptive algorithm deciding PASSIVE
> correctly on the finite family {u₀} ∪ {u₀ + p_j}, j = 1…M = Θ(N), must make ≥ M = Θ(N) queries.
> **Corollary 3 (Yao, stands).** Ω(N) for bounded-error randomized.
> **Corollary 4 (stands).** Two FFTs read N ⇒ query complexity is **Θ(N)**, tight.

**Under these hypotheses, and no others:** the admissible class must contain perturbations supported
on O(1) grid points. It must **not** carry an N-independent bound on ‖p′‖_∞, nor a bandwidth promise
(the document's own §3.5.1). Under either of those, this construction gives only Ω(1) in N and the
theorem is silent.

**§4 / C31 (the V2b barrier) is untouched by this dispute** — it is a literature argument about
Ben-Or and Björner–Lovász–Yao and does not use Lemma 1.

## 9. Recommendation for CLAIMS.md

- **C30 → restore to `supported`, reworded** as §8 above (κ = 1.254 = 1/h_φ, h_φ = 0.79758; ball
  radius 1.68·D; the roughness hypothesis stated in the headline, not buried in §3.5). It has now
  been reproduced by an independent hand (this one) with a different observable, two bump families,
  two grid conventions, and an analytic cross-check of the constant. It is **not** yet `replicated`
  in the crew's sense — I am red team, not the blind replicator.
- **C37 → mark RESOLVED, not upheld.** Record: the parity fact is true; the factor 2.6, the 15 %
  drift and the 1.64× x₀-spread are all one artifact — probing at the bump edge (σ ≈ −1) instead of
  the extremum (σ* = −0.723). Its N-independence result stands and corroborates cycle 2.
- **Process note:** the replicator was dispatched with a paraphrase, not a quotation. The paraphrase
  dropped the words "min_x". That single omission cost a cycle. Blind replication must be dispatched
  with the *verbatim* formal statement and the *verbatim* definition of the measured observable,
  even when the source files are withheld.
