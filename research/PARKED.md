# Parked — the flipped optimization problem

**Status: deliberately NOT pursued in this voyage.** Parked by Captain's decision, 2026-08-18.

## What it is

Invert the map: start from a loss profile ε″ = v ≥ 0 (passive **by construction**), recover its sKK
partner ε′ via Thm 6.1, and optimize reflectance over the resulting design space.

Wave 3 already carries this a long way: Thm 6.1 (complete parametrization of the passive set:
ε′ = c − (D/2L)x − ℋ[v], with the passive set the image of the convex cone {v ≥ 0, v(±L) = 0}),
Cor 6.2 (Fejér–Riesz: band-limited passive designs are exactly v = (1 + cos(πx/L))|Q|²), the
budget–performance frontier table (a two-step profile matches the logistic's six-order suppression
with **half** the absorption budget), and `constructor2.py`.

## Why it is parked, not pursued

It is a **generation** problem: produce passive profiles. The mission is a **decision** problem:
given an ε′ you already care about, is it passive? Thm 6.1 gives you passive profiles; it does not
tell you whether a *given* ε′ is one of them. Conflating the two is exactly how the Apr–Aug sessions
drifted off the Captain's original question.

## Why it is worth its own voyage later

Wave 3's own Open problem 8.2 names it "The frontier paper": map the (σ, R(λ,θ)) Pareto set over the
passive cone at fixed (D, L) — LP/SDP inner loop, TMM outer loop, compared against the
equilibrium-measure boundary. That is a complete, separable second paper with its own mission and
its own crew.

## Rules of engagement

- Thm 6.1 / Cor 6.2 may be **used as tools** — e.g. to generate labelled passive test profiles for
  the adversarial crew, or as a source of near-threshold adversarial cases. That is not drift.
- Optimizing reflectance over the passive cone **is** drift. If a cycle starts heading there, the
  First Mate stops it and logs the temptation here.

## Update 2026-08-19 — this voyage built the parked problem's tool

Navigator's finding: the frontier optimization no longer needs an LP/SDP inner loop. **(L2), the
extreme-point reduction, plus Bauer's maximum principle gives closed-form extreme points**, which is
exactly the expensive inner step Open Problem 8.2 assumed would need convex programming. The parked
voyage is now materially cheaper than when it was parked — worth telling the Captain when he chooses
the next mission.
