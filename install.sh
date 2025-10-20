#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/Source"

if [[ ! -d "${SOURCE_DIR}" ]];
then
    echo "Source directory not found: ${SOURCE_DIR}" >&2
    exit 1
fi

echo "Installing JackFramework from ${SOURCE_DIR}"
python -m pip install --upgrade --editable "${SOURCE_DIR}"
echo "Install finished"
