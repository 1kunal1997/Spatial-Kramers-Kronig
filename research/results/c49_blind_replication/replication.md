# Blind replication report — Claim C49 (board item F5)

**Role:** replicator (blind). **Input:** `research/results/c49_blind_replication/witness_input.json`.
**Forbidden material:** did not open, read, grep, or execute anything under
`research/results/fable_ns_answer/` or `research/theory/ns_answer.md`. Confirmed clean.

## VERDICT: **MATCH**

Both independent routes agree with each other and with the claim as restated:
weight vector **A is GAIN**, weight vector **B is PASSIVE**, both share identical
total mass, identical M1 = 0, and identical readings on all 8 battery functionals
to floating-point precision.

---

## Route used, and why it is independent

**Primary route (Rank 1 — analytic closed form).** The DiHiTI construction for an
atomic drop measure g = Σ w_j δ(t−t_j) admits an *exact* closed form, not just a
numerical algorithm. It is the logarithmic-potential master identity quoted
verbatim in `research/MISSION.md` (a document I am permitted to read, listed as
foundational assumption #3 of this mission, independently derived and dated
2026-08-13 — well before this witness was generated):

```
eps''(x) = (1/pi) * ∫ g(t) * ln[ cos(pi t / 2L) / |sin(pi (x-t)/2L)| ] dt
```

For an atomic measure this collapses to a finite sum with **no FFT, no grid, no
smoothing, no endpoint-handling choice at all**:

```
eps''(x) = (1/pi) * Σ_j w_j * ln[ cos(pi t_j/2L) / |sin(pi (x-t_j)/2L)| ]
```

This is the strongest available route per the replicator ranking (analytic
closed form beats "different numerical method" beats "different
discretization"), and it is structurally independent of `ht_derivative`'s
FFT-based derivative→Hilbert→integrate→de-tilt pipeline — it shares no code,
no grid, and no library call with it.

**Secondary route (cross-check, different discretization of the named
production pipeline).** The dispatch brief explicitly names
`tmm_helper.ht_derivative` as "production" for the DiHiTI map, so it was also
run directly, as a check on the analytic route rather than as the sole
evidence: atoms were realized as a smoothed decreasing staircase (erf
transitions of width `s`) on a **cell-centered, non-duplicating grid**
`x = -L + 2L*(arange(N)+0.5)/N` (chosen specifically to avoid the
`np.linspace(-L,L,N)` endpoint-duplication bug flagged in `PROTOCOL.md`'s C22
post-mortem), swept over `N ∈ {4001, 6000, 8192, 16001}` (mixing power-of-two
and non-power-of-two) and `s ∈ {0.02, 0.01, 0.005, 0.002}`.

Code: `research/results/c49_blind_replication/replication/`
- `1_analytic_route.py` — closed-form route, battery, interior-min scan
- `2_diagnostic_scan.py` — fine full-domain scan (2e6 points) confirming sign structure
- `3_ht_derivative_route.py` — `tmm_helper.ht_derivative` cross-check, N/s sweep
- `4_control_test.py` — independent sanity control (below)

---

## Harness control (asserted before trusting any witness number)

Per the dispatch brief's suggested control: a single balanced atomic pair at
`t = -a, +b` (both weights positive) with `M1 = -w_a tan(pi a/2) + w_b tan(pi b/2) = 0`
enforced is known analytically to be **passive iff a+b ≤ 1**. This is
independent of the witness data entirely — pure sanity check on the harness.

Both routes were run on a sweep of `a+b` from 0.6 to 1.4 (symmetric `a=b` pairs
and asymmetric pairs with `M1=0` enforced via `w_b = w_a·tan(πa/2)/tan(πb/2)`):

| a+b | analytic min ε″ | verdict | ht_derivative min ε″ | verdict |
|---|---|---|---|---|
| 0.60 | 0.000000 | PASSIVE | +0.000401 | PASSIVE |
| 0.90 | 0.000000 | PASSIVE | +0.000550 | PASSIVE |
| 0.96 | 0.000000 | PASSIVE | +0.000599 | PASSIVE |
| **1.00** | 0.000000 | boundary | −0.000000 | boundary |
| 1.04 | −0.040026 | GAIN | −0.040031 | GAIN |
| 1.20 | −0.203373 | GAIN | −0.203400 | GAIN |
| 1.40 | −0.429257 | GAIN | −0.429328 | GAIN |

Asymmetric pairs with `M1=0` enforced (e.g. `a=0.30, b=0.75` → `a+b=1.05` →
GAIN; `a=0.30, b=0.60` → `a+b=0.90` → PASSIVE) reproduce the same `a+b≤1`
threshold. **Both routes cross the threshold at exactly the right point, with
magnitudes agreeing to ≤0.03% for `a+b>1`.** Harness trusted.

Full sweep output in `research/results/c49_blind_replication/replication/4_control_test.py`
(run it directly to reproduce the table above).

---

## Numbers

**Agreement tolerance stated before computation (my own, since this claim has
no source number to compare to a stated tolerance — I am determining sign, not
matching a numeric target):** PASSIVE/GAIN sign must be unambiguous and stable
(margin ≫ discretization noise, i.e. not a coin-flip near zero) and must not
flip across grid resolution `N`, smoothing width `s`, or the analytic-vs-grid
route.

### 1. PASSIVE/GAIN determination

| | verdict | interior min ε″ | location x | negative-region size |
|---|---|---|---|---|
| **weights_A** | **GAIN** | −0.007756 (converged n=20001→500001; converged across all N,s in route 2, range −0.00696 to −0.00775) | x ≈ 0.6311 | x ∈ [0.613, 0.649], ~1.8% of interior points negative (2e6-point full scan) |
| **weights_B** | **PASSIVE** | +2.24e-13 at margin 1e-7 (i.e. 0 to numerical precision, approached only at the edges) | x ≈ −0.9999999 (edge) | 0% of interior points negative (2e6-point full scan) |

Interior-minimum convention used (stated deliberately per the harness-trap
warning): dense analytic evaluation on (−1+margin, 1−margin) excluding a small
radius around each atom (where the log-potential has an integrable +∞
singularity — always a **local max** for positive weights, never a spurious
min, and confirmed excluded correctly by the 0%-negative / 1.8%-negative scan
above). The endpoint value → 0 was verified *separately* as a check on the
formula (edge margins 1e-2, 1e-4, 1e-6 all shrink toward 0 correctly for both
A and B) and explicitly **not** used as the passivity indicator, per the
blind-indicator warning in the brief.

### 2. Battery readings

φ₁=mass, φ₂=M1, φ₃=M2, φ₄=σ, φ₅=t, φ₆=t², φ₇=t³, φ₈=t⁴ — computed as exact
sums over the 20 atoms (no grid, no discretization risk, and no `tan`
overflow since the nearest atom to the boundary is at t=0.95, giving
tan(π·0.95/2) ≈ 12.7, far from the 1.6e16 overflow the brief warns about):

| φ | A | B | abs diff |
|---|---|---|---|
| φ₁ (mass) | 19.6310337 | 19.6310337 | 0 |
| φ₂ (M1) | 2.22e-16 | 4.44e-16 | 2.2e-16 |
| φ₃ (M2) | 57.1988150 | 57.1988150 | 7.1e-15 |
| φ₄ (σ) | 9.4915809 | 9.4915809 | 1.8e-15 |
| φ₅ | −2.4839200 | −2.4839200 | 8.9e-16 |
| φ₆ | 2.6023264 | 2.6023264 | 4.4e-16 |
| φ₇ | 0.6233508 | 0.6233508 | 0 |
| φ₈ | 0.9336787 | 0.9336787 | 3.3e-16 |

**Max absolute difference: 7.1e-15** — consistent with float64 roundoff at the
magnitudes involved (≤57), i.e. **identical to machine precision**, confirming
the "identical readings under all 8 functionals" and "M1=0 exactly" parts of
the claim. (Relative differences are not meaningful for φ₂, φ₇ since both
values are ≈0 or the diffs are exactly 0; the absolute-difference table is the
correct comparison here and it shows agreement at the roundoff floor.)

---

## Convergence evidence

- **Analytic route:** interior-min search re-run at grid densities
  n = 20001, 50001, 200001, 500001 for the min-search sampling grid — value
  and location stable to the displayed precision (−0.007756 @ x=0.6311 for A;
  ≈0 @ edge for B) across all four.
- **ht_derivative route:** verdicts stable across **all 16** combinations of
  N ∈ {4001, 6000, 8192, 16001} (2 non-powers-of-two, 2 powers-of-two/near) and
  s ∈ {0.02, 0.01, 0.005, 0.002} — no flip anywhere for either A or B. Minimum
  magnitude for A converges from −0.00696 (coarse smoothing s=0.02) to
  −0.00775 (fine smoothing s=0.002), i.e. converging *toward* the analytic
  value as smoothing width shrinks, as expected.
- **Cross-route agreement:** grid-based `ht_derivative` output (N=8192, s=0.01)
  matches the analytic closed form to **1.2% relative RMS**, masked away from
  atom singularities and the edge margin — the residual is attributable to
  finite smoothing width `s` and finite grid spacing, both of which shrink the
  discrepancy when refined (confirmed above).

---

## Labels resolved

- **weights_A → GAIN**
- **weights_B → PASSIVE**

## Verdict details

**MATCH.** The claim as restated — 20 shared atoms, two positive weight
vectors, identical total mass, M1=0 exactly, identical readings under 8 linear
functionals, yet one profile PASSIVE and the other GAIN — is reproduced by an
independent analytic closed-form route (no FFT, no grid, no smoothing) and
cross-checked by the named production pipeline at multiple resolutions and
smoothing widths, with a control against a known-closed-form passivity
threshold (a+b≤1 for a balanced pair) passing on both routes. No mismatch was
found; nothing was tuned to make it agree — the closed-form route settled on
its answer before the ht_derivative route was run at all.
