#!/bin/bash

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$(pwd):$PYTHONPATH"
python tools/verify_all.py "$@"
