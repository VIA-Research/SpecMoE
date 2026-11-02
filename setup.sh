#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash setup.sh [--cpu]

Installs the pinned dependencies into the active Python 3.9 environment.
The default installs PyTorch with CUDA 12.4; --cpu installs the CPU build
for tests. Set PYTHON to select a Python interpreter explicitly.

Create the reference environment first:
  conda create -n specmoe python=3.9.23
  conda activate specmoe
  bash setup.sh
EOF
}

torch_build=cu124
if (( $# > 1 )); then
    usage >&2
    exit 2
fi
case ${1:-} in
    "") ;;
    --cpu) torch_build=cpu ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON:-python}

"${python_bin}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 9):
    raise SystemExit("Python 3.9 is required; create the reference environment with Python 3.9.23.")
PY

"${python_bin}" -m pip install --upgrade pip
"${python_bin}" -m pip install "torch==2.5.0+${torch_build}" \
    --index-url "https://download.pytorch.org/whl/${torch_build}"
"${python_bin}" -m pip install -r "${script_dir}/requirements.txt"
"${python_bin}" -m pip check
