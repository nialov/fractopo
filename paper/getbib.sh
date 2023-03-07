#!/usr/bin/env bash

set -euxo pipefail

pandoc -L getbib.lua paper.md -o refs.bib
