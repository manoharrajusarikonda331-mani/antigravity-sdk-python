#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Release script for the Google Antigravity SDK.
#
# Builds platform-specific wheels containing the pre-compiled Go
# localharness binary and uploads them to the OSS Exit Gate for
# distribution.
#
# This script works in two modes:
#   1. Kokoro: Runs inside a Kokoro release job. Binaries are fetched
#      from MPM via Kokoro's fetch_mpm (pre-populated by Rapid).
#   2. Local: Run from a Copybara export directory after manually
#      placing binaries under .kokoro/binaries/<platform>/localharness.
#
# Environment variables:
#   VERSION         - SDK version (default: auto-read from pyproject.toml).
#
# Usage (local, after Copybara export):
#   # Place binary(ies) under .kokoro/binaries/<platform>/localharness,
#   # then run:
#   VERSION=0.1.1 .kokoro/release.sh
#
# Usage (single platform, local shortcut):
#   mkdir -p .kokoro/binaries/linux-x86_64
#   cp /path/to/localharness .kokoro/binaries/linux-x86_64/localharness
#   .kokoro/release.sh

set -eo pipefail

# --- Resolve SCRIPT_DIR before cd so the relative BASH_SOURCE path is
# resolved against the original working directory (the Kokoro workspace root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Determine project root ---
if [[ -n "${KOKORO_ARTIFACTS_DIR}" ]]; then
  cd "${KOKORO_ARTIFACTS_DIR}/git/antigravity-sdk-py"
fi

# If a specific GoB commit was requested (via GOB_COMMIT build param),
# check it out to pin the Python source to a specific Piper CL.
# Kokoro exposes build_params as environment variables.
if [[ -n "${GOB_COMMIT:-}" ]]; then
  echo "--- Pinning SDK source to GoB commit: ${GOB_COMMIT} ---"
  git checkout "${GOB_COMMIT}"
fi

PROJECT_DIR="$(pwd)"

# --- Python 3.13 via pyenv (pre-installed on the Kokoro image) ---
echo "--- Setting up Python 3.13 ---"
eval "$(pyenv init -)"
pyenv install -s 3.13
pyenv global 3.13
python3 --version

# --- Read version from pyproject.toml if not set ---
if [[ -z "${VERSION}" ]]; then
  VERSION=$(python3 -I -c "import tomllib, pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])")
fi

echo "=== Google Antigravity SDK Release v${VERSION} ==="

# Install build/release tools with hash verification.
# See go/pip-install-remediation.
python3 -m pip install \
  --require-hashes \
  -r "${SCRIPT_DIR}/requirements-release.txt"

DIST_DIR="dist"
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

# --- Platform definitions ---
declare -A PLATFORM_TAGS=(
  ["linux-x86_64"]="manylinux_2_17_x86_64"
  ["linux-arm64"]="manylinux_2_17_aarch64"
  ["darwin-arm64"]="macosx_11_0_arm64"
)

BINARY_NAME="localharness"
BIN_DEST="google/antigravity/bin"
BINARIES_DIR=".kokoro/binaries"

# --- MPM directory mapping (populated by Kokoro fetch_mpm) ---
MPM_DIR="${KOKORO_ARTIFACTS_DIR:-}/mpm"
declare -A MPM_DIRS=(
  ["linux-x86_64"]="localharness_linux_x86_64"
  ["linux-arm64"]="localharness_linux_arm64"
  ["darwin-arm64"]="localharness_darwin_arm64"
)

# --- Fetch binaries from MPM or local ---
for PLATFORM in "${!PLATFORM_TAGS[@]}"; do
  LOCAL_BIN="${BINARIES_DIR}/${PLATFORM}/${BINARY_NAME}"
  if [[ ! -f "${LOCAL_BIN}" ]]; then
    MPM_SUBDIR="${MPM_DIRS[$PLATFORM]:-}"
    MPM_BIN="${MPM_DIR}/${MPM_SUBDIR}/localharness_external"
    if [[ -n "${MPM_SUBDIR}" && -f "${MPM_BIN}" ]]; then
      echo "--- Copying ${PLATFORM} binary from MPM ---"
      mkdir -p "${BINARIES_DIR}/${PLATFORM}"
      cp "${MPM_BIN}" "${LOCAL_BIN}"
      chmod +x "${LOCAL_BIN}"
    else
      if [[ -n "${KOKORO_ARTIFACTS_DIR}" ]]; then
        echo "ERROR: No binary for ${PLATFORM} (looked in ${MPM_BIN})."
        echo "In a release job, all platform binaries must be available."
        exit 1
      fi
      echo "WARNING: No binary for ${PLATFORM} (looked in ${MPM_BIN}), skipping."
      continue
    fi
  fi
done

# --- Build platform-specific wheels ---
BUILT_ANY=false

for PLATFORM in "${!PLATFORM_TAGS[@]}"; do
  WHEEL_PLAT="${PLATFORM_TAGS[$PLATFORM]}"
  LOCAL_BIN="${BINARIES_DIR}/${PLATFORM}/${BINARY_NAME}"

  if [[ ! -f "${LOCAL_BIN}" ]]; then
    echo "--- Skipping ${PLATFORM}: no binary available ---"
    continue
  fi

  echo "--- Building wheel for ${PLATFORM} (${WHEEL_PLAT}) ---"

  # Place the binary into the package namespace.
  mkdir -p "${BIN_DEST}"
  cp "${LOCAL_BIN}" "${BIN_DEST}/"
  chmod +x "${BIN_DEST}/${BINARY_NAME}"

  # Ensure __init__.py exists for the bin subpackage so setuptools
  # discovers it via package-data.
  touch "${BIN_DEST}/__init__.py"

  # Build the wheel, then re-tag with the correct platform.
  # --no-isolation: use the setuptools/wheel already installed via
  # requirements-release.txt rather than creating a fresh venv that
  # downloads from PyPI (which can time out and bypasses hash verification).
  python3 -m build --wheel --no-isolation --outdir "${DIST_DIR}"
  python3 -m wheel tags \
    --platform-tag="${WHEEL_PLAT}" \
    --remove \
    "${DIST_DIR}"/*-py3-none-any.whl

  echo "  -> $(ls -1 "${DIST_DIR}"/*"${WHEEL_PLAT}"*.whl 2>/dev/null | tail -1)"

  # Clean the binary for the next platform iteration.
  rm -rf "${BIN_DEST}"
  BUILT_ANY=true
done

if [[ "${BUILT_ANY}" != "true" ]]; then
  echo "ERROR: No wheels were built. Ensure binaries are available."
  exit 1
fi

echo ""
echo "--- Built wheels ---"
ls -lh "${DIST_DIR}/"

# --- Upload to OSS Exit Gate ---
REPO_URL="https://us-python.pkg.dev/oss-exit-gate-prod/google-antigravity--pypi/"
echo ""
echo "--- Validating wheels ---"
twine check "${DIST_DIR}"/*
echo ""
echo "--- Uploading to OSS Exit Gate (${REPO_URL}) ---"
twine upload \
  --repository-url "${REPO_URL}" \
  --verbose \
  "${DIST_DIR}"/*

echo "--- Release v${VERSION} complete ---"
