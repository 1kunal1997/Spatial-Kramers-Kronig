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
