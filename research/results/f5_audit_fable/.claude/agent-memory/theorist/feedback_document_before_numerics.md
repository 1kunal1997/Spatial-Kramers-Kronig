---
name: feedback-document-before-numerics
description: Write the proof document to disk before running optional/confirmatory numerics; a verified-but-unwritten proof is worth nothing if the session is terminated
metadata:
  type: feedback
---

Write the deliverable document **first**, or at least before any optional extra checks. Run only the
numerics that the document cannot be written without; append the rest afterwards using a
`**RESULTS-PLACEHOLDER**` marker in the draft.

**Why:** this voyage runs on a metered plan and sessions are terminated without warning mid-task (it
happened during F1 on 2026-08-19). The proof is the deliverable for the victory condition; numbers
are only checks on it. A verified-but-unwritten proof is worth nothing.

**How to apply:** draft the full argument with a placeholder in the verification section, checkpoint,
then run scripts and splice results in. Also: the coordinator may have already verified part of your
construction while you were down — read the resume message before rebuilding a script you planned.
