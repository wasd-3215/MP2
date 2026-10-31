# Answer-group controlled experiment, 2026-10-18

Protocol is frozen before fitting in `protocol.json`. All scripts operate only in this experiment directory. Original official bank, parser, runner and scorer are unchanged.

Run with the existing mp1 Python, bytecode disabled: `check_models.py`, `constraint_audit.py`, `train.py`, `evaluate_suite.py`, `analyze.py`, `verify.py`, in that order. Numerical jobs run serially with four Torch threads. Training never loads validation. `results/REPORT.md` records results and caveats after evaluation.

The six portable directories are six alternative predictors, not a single combined inference package. Deploy only one cell with its own method and pipeline. Each cell is checked against the 16 MiB expanded asset budget. Historical 740 is contextual; primary conclusions compare the newly matched cells.
