#!/usr/bin/env bash
# Start the generator task after beads 861 and zud have both been merged.
# This task depends on both foundation tasks and should not run earlier.
set -euo pipefail

workmux add feat/porepy-generator -b -p \
  'Implement bead feat-add-porepy-interoperability-w0b. Read the bead with `br show feat-add-porepy-interoperability-w0b --json`, claim it with the required actor, and follow its acceptance criteria exactly. Confirm predecessor beads 861 and zud are closed and their changes are present before editing. Implement the minimal Network-to-random-circular-fracture CSV exporter, with its focused tests. Run the specified focused pytest and prek checks. Close the bead only with test evidence, run br sync --flush-only, and commit the source/tests plus forced .beads/issues.jsonl with --no-gpg-sign and the bead id. Do not start the documentation/final-validation bead.'

printf '%s\n' 'Started generator task w0b.'
printf '%s\n' 'After it is merged, run start_porepy_finalize.sh.'
