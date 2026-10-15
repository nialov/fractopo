#!/usr/bin/env bash
# Start the first PorePy implementation task in an isolated worktree.
#
# 861 and zud both edit fractopo/interop/porepy.py and
# tests/interop/test_porepy.py, so they are intentionally NOT started in
# parallel: doing so creates an avoidable merge conflict.
set -euo pipefail

workmux add feat/porepy-valid-samples -b -p \
  'Implement bead feat-add-porepy-interoperability-861. Read the bead with `br show feat-add-porepy-interoperability-861 --json`, claim it with the required actor, and follow its acceptance criteria exactly. Work only on the valid Network fracture-set extraction task. Run the specified focused pytest and prek checks. Close the bead only with test evidence, run br sync --flush-only, and commit the source/tests plus forced .beads/issues.jsonl with --no-gpg-sign and the bead id. Do not start dependent beads.'

printf '%s\n' 'Started foundation task 861.'
printf '%s\n' 'After it is merged, run start_porepy_semiaxis.sh.'
