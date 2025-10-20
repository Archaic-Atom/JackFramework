#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/Source"
DIST_DIR="${SOURCE_DIR}/dist"

if [[ ! -d "${SOURCE_DIR}" ]];
then
    echo "Source directory not found: ${SOURCE_DIR}" >&2
    exit 1
fi

if ! python -m build --version >/dev/null 2>&1;
then
    echo "Python package 'build' is required. Install it via 'python -m pip install build'." >&2
    exit 1
fi

echo "Building JackFramework distributions from ${SOURCE_DIR}"
rm -rf "${DIST_DIR}"
python -m build "${SOURCE_DIR}"

echo "Build artifacts stored in ${DIST_DIR}"
