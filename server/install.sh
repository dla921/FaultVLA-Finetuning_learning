#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -c 'import sys; assert sys.version_info[:2] == (3,10), "Activate a Python 3.10 environment"'
python -m pip install -c constraints.txt -r requirements-lock.txt
python -m pip install --no-deps -e ./VLA-Adapter
python -m pip check
