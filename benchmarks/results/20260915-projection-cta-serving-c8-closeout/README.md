# C8 user closeout

See [final comparison](../../../deploy/260913/29-serving-closeout.md). This archive is a user-stopped, nine-lane run: shared crossed-order comparison is complete; unique is single-order only. It is not a completed 12-lane matrix or overall performance qualification.

Run `python3 benchmarks/results/20260915-projection-cta-serving-c8-closeout/verify_closeout.py` from the repository root. The verifier validates all completed rows and lifecycle interruption; it intentionally requires the user-closeout marker and rejects a complete-matrix claim.
