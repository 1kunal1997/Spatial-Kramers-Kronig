# Captain's Queue

Non-blocking questions and observations for Kunal, accumulated between checkpoints.
**Nothing here stops work.**

| Date | Item | Why it matters | Assumed answer used meanwhile |
|------|------|----------------|-------------------------------|
| 2026-08-18 | **Durable auto-resurrection needs the Captain's hands.** Local cron is session-only (dies with the session). Cloud routines are genuinely durable but run on a **GitHub checkout**, and (a) GitHub App access to `1kunal1997/Spatial-Kramers-Kronig` could not be verified, (b) key mission inputs are **untracked**: `verify_passivity.py`, the Aug-18 PDF notes, and the whole labelled figure corpus (`*.pdf`/`*.png` are gitignored). A cloud agent would clone a repo missing the mission's core evidence. | Determines whether the voyage can continue with the laptop closed. | Proceeding with **local + ledger durability**: all state is on disk and committed, so any new session resumes from `PROTOCOL.md`. Auto-resurrection deferred. |
| 2026-08-18 | Should `verify_passivity.py` (and possibly the Aug-13/Aug-18 PDFs + the Apr/May figure corpus) be committed? | Untracked evidence cannot be replicated by a cold session, and `verify_passivity.py` is the reference implementation for half the inherited claims. | Treating them as read-only local inputs; not committing without the Captain's say-so (`*.pdf`/`*.png` are gitignored deliberately). |
