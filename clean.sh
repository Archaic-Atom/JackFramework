#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/Source"

echo "Removing build artifacts"
rm -rf "${SOURCE_DIR}/build" "${SOURCE_DIR}/dist" "${SOURCE_DIR}/JackFramework.egg-info"

find "${SCRIPT_DIR}" -type f \( -name "*.log" -o -name "*.pyc" \) -delete
find "${SCRIPT_DIR}" -type d -name "__pycache__" -prune -exec rm -rf {} +

echo "Clean finished"
