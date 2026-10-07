#!/bin/bash
# Copyright Axelera AI, 2026
# Install the AxScript kernel test environment into the ACTIVE python interpreter.
#
#     tools/install_kernel_env.sh [--sp-revision <sha|branch>]
#
# One command, so that CI and model_axscript/internal/kernel_dev_setup.md install the same
# environment rather than two hand-kept copies of it.
#
# This repository states ONE version, the software-platform revision below. Every component
# version is resolved from it by tools/resolve_kernel_env.py: the devkit and rt wheels built
# from that revision, the compiler wheels built from the host.compiler.alto revision its
# west.yml pins, and the simulator wheel built from the riscv-isa-sim revision it pins. Nothing
# here is a hand-picked component version, because software-platform is what decides which
# compiler compiles which framework tree, and west.yml is where it says so.
#
# Python packages only. device-tree-compiler is a system package that a kernel compile shells
# out to, and the caller installs it: apt in CI, the rootless unpack in containerless.sh.
set -euo pipefail

here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
resolve="${here}/resolve_kernel_env.py"

# The software-platform revision this tree is tested against. Bump this ONE line to move the
# whole environment; the nightly passes `main` instead, so the day software-platform moves past
# it is reported by that lane rather than by somebody's unrelated pull request going red.
sp_revision=6d1f03753244441d53da71c9fc24cb197b26a786
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sp-revision) sp_revision="$2"; shift 2 ;;
        *) echo "install_kernel_env.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

echo "Resolving the kernel environment from software-platform ${sp_revision}"
python "${resolve}" --print-revisions --sp-revision "${sp_revision}"

# THREE resolves, in this order, and not one. axelera-devkit depends on the compiler at its own
# bare release label, so asking for the devkit and a post-release compiler together is
# ResolutionImpossible; the second install overrides what the devkit brought. The simulator is
# last because it depends on neither.
for part in devkit compiler simulator; do
    python "${resolve}" --part "${part}" --sp-revision "${sp_revision}" \
        | python -m pip install \
            --index-url https://artifactory.production.aws.axelera.ai/artifactory/api/pypi/swci-internal-pypi/simple \
            --trusted-host artifactory.production.aws.axelera.ai \
            -r /dev/stdin
done
