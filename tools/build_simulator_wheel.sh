#!/usr/bin/env bash
# Copyright Axelera AI, 2026
#
# Build a RELOCATABLE ax-accel-sim wheel from the software-platform source that west pins,
# and optionally install it into the active environment.
#
# Why this exists. The AICore functional simulator ships inside the ax-accel-sim wheel, and
# a wheel built by a plain `setup.py bdist_wheel` is not usable off the machine that built
# it: nothing vendors the libraries its spike plugins load with dlopen, and the platform tag
# claims any linux while the binaries need the glibc of the build host. `auditwheel repair`
# is what fixes both, and the package already asks for it, in
# `axelera/python/ax-accel-sim/pyproject.toml`:
#
#     repair-wheel-command = "LD_LIBRARY_PATH=... auditwheel repair --plat manylinux_2_35_$(uname -m) ..."
#
# So this script writes no packaging of its own. It runs the package's own cibuildwheel
# configuration against the revision west pins, which is what makes the result reproducible:
# two people on two machines build the same wheel from the same pin.
#
# Prefer a published wheel when one exists. Build only when the version you need is not
# published as a repaired manylinux wheel.
#
# Usage:
#   tools/build_simulator_wheel.sh [--install] [--python 3.12] [--output-dir DIR]
#
# Run it ON THE HOST, not inside `make homeless`: cibuildwheel starts its own containers, so
# running it inside the dev container would nest docker in docker. Activate whichever venv you
# want `--install` to land in first.
#
# Requires docker, a one-time `make helper-venv` in software-platform, and about 15 minutes.

set -euo pipefail

_install=false
_python="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
_out=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) _install=true; shift ;;
        --python) _python="$2"; shift 2 ;;
        --output-dir) _out="$2"; shift 2 ;;
        -h|--help) sed -n '3,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# Find the enclosing software-platform checkout the same way containerless.sh does: by the
# west.yml that marks it. Walking up from this script is not enough on its own, because a git
# WORKTREE of the framework lives outside that tree entirely, so fall back to the main
# checkout that the shared git common dir points at.
_this=$(readlink -f "$(dirname -- "$0")")

# One marker check, used from two starting points. Spelled twice it had already drifted.
_walk_up_to_software_platform() {
    local candidate="$1"
    while [[ -n "${candidate}" && "${candidate}" != "/" ]]; do
        if [[ -f "${candidate}/west.yml" && -d "${candidate}/host/application/framework" ]]; then
            echo "${candidate}"
            return 0
        fi
        candidate=$(dirname "${candidate}")
    done
    return 1
}

# From this script, then, for a worktree that lives outside the tree entirely, from the git
# common dir, which points back into the main checkout.
_software_platform=$(_walk_up_to_software_platform "${_this}") \
    || _software_platform=$(_walk_up_to_software_platform \
        "$(git -C "${_this}" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)") \
    || _software_platform=""
if [[ -z "${_software_platform}" ]]; then
    echo "ERROR: no software-platform checkout found from ${_this} (looked for a west.yml" >&2
    echo "       beside host/application/framework, and through the git common dir)." >&2
    exit 1
fi
_sim="${_software_platform}/tools/simulator/riscv-isa-sim"
_out="${_out:-${_software_platform}/wheelhouse}"

if [[ ! -d "${_sim}/axelera/python/ax-accel-sim" ]]; then
    echo "ERROR: no simulator source at ${_sim}." >&2
    echo "       It is a west project; run 'west update' in ${_software_platform} first." >&2
    exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: cibuildwheel builds inside a container and docker is not on PATH." >&2
    exit 1
fi

# Report the revision, and whether it is the pinned one. A wheel built from a dirty or
# moved checkout is not the artifact anyone else would get from the same pin, and silently
# shipping one is how a 'reproducible' build stops being reproducible.
_head=$(git -C "${_sim}" rev-parse HEAD 2>/dev/null || echo "unknown")
_pinned=$(python3 - "${_software_platform}/west.yml" <<'PY' 2>/dev/null || echo "unknown"
import re, sys
text = open(sys.argv[1]).read()
block = re.search(r"- name: riscv-isa-sim\b.*?(?=\n    - name: |\Z)", text, re.S)
print(re.search(r"revision:\s*(\S+)", block.group(0)).group(1) if block else "unknown")
PY
)
echo "simulator source: ${_sim}"
echo "  HEAD:      ${_head}"
echo "  west pin:  ${_pinned}"
if [[ "${_head}" != "${_pinned}" && "${_pinned}" != "unknown" ]]; then
    echo "  WARNING: HEAD is not the pinned revision, so this wheel is not what the pin builds." >&2
fi
if [[ -n "$(git -C "${_sim}" status --porcelain 2>/dev/null)" ]]; then
    echo "  WARNING: the simulator checkout is dirty; the wheel will not be reproducible." >&2
fi

# The base image tag is a content hash of the simulator repo's docker dir, so it cannot be
# pinned here and is resolved the way that repo's own CI resolves it: through repo-cli.py,
# which needs the `docker_wrapper` module that software-platform's helper venv provides.
#
# Called by its interpreter rather than by activating that venv, because --install below has
# to land in the environment the caller is using, not in the helper venv.
_helper_python="${_software_platform}/venv/bin/python"
if [[ ! -x "${_helper_python}" ]]; then
    echo "ERROR: no helper venv at ${_software_platform}/venv." >&2
    echo "       Create it once with:  cd ${_software_platform} && make helper-venv" >&2
    echo "       Do NOT run this script inside 'make homeless': cibuildwheel starts its own" >&2
    echo "       containers, so it belongs on the host." >&2
    exit 1
fi

echo "resolving the cibuildwheel base image..."
cd "${_sim}"
if ! _base=$("${_helper_python}" ./repo-cli.py docker image-url ubuntu_2204 2>/dev/null | tail -1); then
    echo "ERROR: could not resolve the base image url through repo-cli.py." >&2
    exit 1
fi
echo "  base image: ${_base}"
echo "building the cibuildwheel image (pulls from ECR; run 'aws ecr get-login-password' first"
echo "  if docker reports no credentials)..."
_tag="ax-accel-sim-cibuildwheel:${_head}"
docker build --build-arg BASE_IMAGE="${_base}" -t "${_tag}" docker-images/cibuildwheel

echo "building and repairing the wheel for python ${_python}..."
_cibw_venv="${_sim}/.cibw-venv"
[[ -d "${_cibw_venv}" ]] || python3 -m venv "${_cibw_venv}"
"${_cibw_venv}/bin/pip" install -q -r requirements.build.ci.txt

# One interpreter, not the CI matrix: a developer needs the one their venv runs.
CIBW_BUILD="cp${_python//./}-manylinux_*" \
CIBW_MANYLINUX_X86_64_IMAGE="${_tag}" \
CIBW_MANYLINUX_AARCH64_IMAGE="${_tag}" \
    "${_cibw_venv}/bin/cibuildwheel" --platform linux --output-dir "${_out}" \
    axelera/python/ax-accel-sim

_wheel=$(ls -t "${_out}"/ax_accel_sim-*manylinux*.whl 2>/dev/null | head -1 || true)
if [[ -z "${_wheel}" ]]; then
    echo "ERROR: no manylinux wheel in ${_out}. An untagged linux_x86_64 wheel means the" >&2
    echo "       repair step did not run, which is the whole reason to use this script." >&2
    exit 1
fi
echo "built: ${_wheel}"

# Prove the wheel is self-contained before anyone trusts it. An unrepaired wheel passes
# every import test on the build machine and fails on dlopen everywhere else, so the check
# that matters is whether the bundled libraries resolve with nothing else on the path.
python3 - "${_wheel}" <<'PY'
import pathlib, subprocess, sys, tempfile, zipfile

wheel = pathlib.Path(sys.argv[1])
with tempfile.TemporaryDirectory() as tmp:
    zipfile.ZipFile(wheel).extractall(tmp)
    root = pathlib.Path(tmp)
    libs = sorted(root.glob("*.libs"))
    print(f"  vendored library dir: {libs[0].name if libs else 'NONE'}")
    unresolved = {}
    for so in sorted(root.rglob("_sim/lib/*.so")):
        out = subprocess.run(["ldd", str(so)], capture_output=True, text=True).stdout
        missing = [l.split("=>")[0].strip() for l in out.splitlines() if "not found" in l]
        if missing:
            unresolved[so.name] = missing
    if unresolved:
        print("  UNRESOLVED, this wheel is not relocatable:")
        for name, missing in unresolved.items():
            print(f"    {name}: {', '.join(missing)}")
        sys.exit(1)
    print("  every bundled library resolves with nothing extra on the loader path")
PY

if [[ "${_install}" == "true" ]]; then
    echo "installing into $(python3 -c 'import sys; print(sys.prefix)')..."
    python3 -m pip install --force-reinstall --no-deps "${_wheel}"
    echo "done. The simulator entries still need cmake and dtc; see model_axscript/README.md."
else
    echo "not installed. To install it:  python3 -m pip install --force-reinstall --no-deps ${_wheel}"
fi
