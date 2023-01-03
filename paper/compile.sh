#!/usr/bin/env bash

set -euxo pipefail

docker run --rm \
    --volume ./:/data \
    --env JOURNAL=joss \
    openjournals/inara
