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
