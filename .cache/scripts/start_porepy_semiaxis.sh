#!/usr/bin/env bash
# Start the independent semi-axis correction task in an isolated worktree.
# Run start_porepy_foundation.sh first and merge task 861 before using this.
set -euo pipefail

workmux add feat/porepy-semi-axis -b -p \
  'Implement bead feat-add-porepy-interoperability-zud. Read the bead with `br show feat-add-porepy-interoperability-zud --json`, claim it with the required actor, and follow its acceptance criteria exactly. Work only on correcting existing 3D PorePy exporters to use length/2 semi-axes and their focused tests. Run the specified focused pytest and prek checks. Close the bead only with test evidence, run br sync --flush-only, and commit the source/tests plus forced .beads/issues.jsonl with --no-gpg-sign and the bead id. Do not start dependent beads.'

printf '%s\n' 'Started semi-axis task zud.'
printf '%s\n' 'After 861 and zud are merged, run start_porepy_generator.sh.'
