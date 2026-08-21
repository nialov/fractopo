#!/usr/bin/env bash
# Start final tests and documentation after the generator is merged.
set -euo pipefail

workmux add feat/porepy-finalize -b -p \
  'Implement bead feat-add-porepy-interoperability-hkc. Read the bead with `br show feat-add-porepy-interoperability-hkc --json`, claim it with the required actor, and follow its acceptance criteria exactly. Confirm predecessor beads 861, zud, and w0b are closed and their changes are present before editing. Finish only concise exporter documentation and any missing deterministic test coverage; avoid duplicating existing tests or adding speculative APIs. Run focused pytest, full pytest, prek run --all-files, and the conditional Sphinx build specified by the bead. Close only with evidence, run br sync --flush-only, and commit changed files plus forced .beads/issues.jsonl with --no-gpg-sign and the bead id.'

printf '%s\n' 'Started finalization task hkc.'
printf '%s\n' 'After it is merged, close the epic only after its acceptance criteria pass.'
