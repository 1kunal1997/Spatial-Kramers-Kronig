# Expedition duration measurement

**Why:** the Captain is deciding **Pro vs Max**. He needs to know how long a full-throttle
expedition runs before the usage window is exhausted. ~3–4 h → keep Pro. ~1–2 h or less → upgrade
to Max (5×).

## Mechanism

`research/hb.sh <agent> "<note>"` appends one timestamped line to `research/HEARTBEAT.log`.
**Every agent calls it at every 15-minute checkpoint**, and the First Mate calls it at every cycle
boundary. Only a live model can call it, so **the last line records when requests stopped
succeeding** — no counter to drift, no clock to trust, and a skipped checkpoint costs accuracy
rather than correctness.

Duration = last timestamp − the `EXPEDITION START` line.

## Reading the number honestly

**Duration alone is not the answer — burn rate depends on how hard we run.** Four parallel opus/xhigh
hands exhaust a window far faster than one sequential hand. So every expedition logs its
configuration on the START line, and the morning report must quote **duration *and* config together**.
A 4-hour run at one agent and a 1-hour run at four agents are the same burn rate.

## Data point already in hand (cycle 1, 2026-08-18)

**~75% of a 5-hour window in ~25 minutes**, running 4 agents in parallel (2× opus/xhigh, 2× sonnet)
plus one nested hand, including heavy one-time front-loading (3 waves of prior work, two long PDFs,
98 figures, ledger construction).

Extrapolating: a full window ≈ **33 minutes** of 4-way parallel work. Sequential single-agent should
run roughly 4× longer — **~2 hours**, and somewhat more once front-loading is not repeated.

**Provisional read: Pro gives ~1.5–2.5 h of sequential full-throttle research per 5-hour window.**
By the Captain's own threshold that sits in upgrade territory. The overnight run tests it directly.

## Morning report must state

1. Expedition start (UTC) and last heartbeat
2. **Duration before requests began failing**
3. Configuration that produced it (how many hands, which models)
4. Heartbeat count and how many distinct agents reported
5. Recommendation: Pro or Max, against the Captain's ~3 h threshold
