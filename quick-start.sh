#!/usr/bin/env bash
# Copyright Axelera AI, 2026
set -euo pipefail

# Run from the repository root, wherever the script is invoked from
cd "$(dirname "$(readlink -f "$0")")"

say() { echo -e "\033[0;32m==> $*\033[0m"; }

EXTRA_INDEX_URL="${AXELERA_EXTRA_INDEX_URL:-https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple}"

usage() {
    cat <<EOF
Usage: ./quick-start.sh

Installs system dependencies, creates a Python virtual environment in
./venv_<version> with the SDK Python packages, installs that SDK's default
Metis driver, and builds the operators.

The SDK version is read from ./VERSION.TXT.

Options:
  -h, --help  Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'." >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! "$EXTRA_INDEX_URL" =~ ^(https?|file):// ]]; then
    echo "Error: AXELERA_EXTRA_INDEX_URL '${EXTRA_INDEX_URL}' must start with https://, http:// or file://" >&2
    exit 1
fi

if [[ ! -f VERSION.TXT ]]; then
    echo "Error: VERSION.TXT not found; cannot determine the SDK version." >&2
    exit 1
fi
SDK_VERSION="$(< VERSION.TXT)"
if [[ ! "$SDK_VERSION" =~ ^[0-9A-Za-z][0-9A-Za-z._+-]*$ ]]; then
    echo "Error: '${SDK_VERSION}' from VERSION.TXT does not look like a version, e.g. 1.8.0" >&2
    exit 1
fi
say "Using SDK version ${SDK_VERSION} from VERSION.TXT"

if [[ -d "venv_${SDK_VERSION}" ]]; then
    say "venv_${SDK_VERSION} already exists — nothing to do. To run again: rm -rf venv_${SDK_VERSION}"
    exit 0
fi


####### INSTALL SECTION #######

# `apt-get update` is done in `install-dependencies.sh`
say "Installing system dependencies (./install-dependencies.sh)"
./install-dependencies.sh
say "Installing python3 and venv support (apt-get install)"
sudo apt-get install -y python3 python3-venv

say "Creating virtual environment venv_${SDK_VERSION}"
python3 -m venv "venv_${SDK_VERSION}"
# shellcheck source=/dev/null
source "venv_${SDK_VERSION}/bin/activate"

say "Installing SDK Python packages (pip install axelera-rt axelera-devkit[all])"
pip install \
  --extra-index-url "$EXTRA_INDEX_URL" \
  "axelera-rt==${SDK_VERSION}" \
  "axelera-devkit[all]==${SDK_VERSION}"

say "Installing the Metis driver (axdevice driver --install)"
axdevice driver --install
say "Loading the Metis driver (modprobe metis)"
sudo modprobe metis || say "The Metis driver is not loaded yet; it will load after a reboot."

say "Building the operators (make operators)"
make operators

say "Done. Activate the environment with: source venv_${SDK_VERSION}/bin/activate"
