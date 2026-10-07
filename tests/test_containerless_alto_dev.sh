#!/bin/bash
# Copyright Axelera AI, 2026
# Checks the --alto-dev editable shadow: which candidate source tree it picks
# (host/tvm since the ALTO repo split, host/lib for a checkout from before it),
# that a missing tree skips rather than aborts, and when it warns on the skew
# between the installed wheel's revision and the local source.
set -euo pipefail

root=$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")
script="${root}/containerless.sh"

# Load just the shadow logic out of containerless.sh, with its collaborators stubbed.
eval "$(sed -n '/^_alto_dev_shadow_pkgs=(/,/^)/p' "${script}")"
eval "$(sed -n '/^_is_tracked_checkout() {/,/^}/p' "${script}")"
eval "$(sed -n '/^_install_alto_dev_shadow() {/,/^}/p' "${script}")"

_opt_alto_dev=true
_installed_version=""
pip() { [[ "$1" == "show" ]] && printf 'Version: %s\n' "${_installed_version}"; }
_pip_install() { printf 'INSTALL %s\n' "$2"; }

# A software-platform tree holding the given package dirs, each its own git repo
# with a tracked file, so the function's `git rev-parse` sees a real revision.
make_sp() {
    local sp rel
    sp=$(mktemp -d)
    for rel in "$@"; do
        mkdir -p "${sp}/${rel}"
        git -C "${sp}/${rel}" init -q
        touch "${sp}/${rel}/pyproject.toml"
        git -C "${sp}/${rel}" add pyproject.toml
        git -C "${sp}/${rel}" -c user.email=t@example.com -c user.name=t commit -q -m init
    done
    echo "${sp}"
}

# The leftover a pre-move checkout keeps: the directory survives but holds only
# build artifacts, so its enclosing repo tracks nothing under it.
make_stale() {
    mkdir -p "$1/dist" "$1/__pycache__"
    touch "$1/dist/pkg-1.0.whl" "$1/__pycache__/mod.pyc"
}

expect() {  # expect <what> <haystack> <needle>
    if [[ "$2" != *"$3"* ]]; then
        echo "FAIL ($1): expected to find '$3' in:" >&2
        echo "$2" >&2
        exit 1
    fi
}

reject() {  # reject <what> <haystack> <needle>
    if [[ "$2" == *"$3"* ]]; then
        echo "FAIL ($1): did not expect '$3' in:" >&2
        echo "$2" >&2
        exit 1
    fi
}

# Post-split layout: both packages come out of the ALTO checkout at host/tvm.
_software_platform=$(make_sp host/tvm/axelera_compiler host/tvm/codegen/axelera-codegen)
out=$(_install_alto_dev_shadow)
expect alto-layout "$out" "INSTALL ${_software_platform}/host/tvm/axelera_compiler"
expect alto-layout "$out" "INSTALL ${_software_platform}/host/tvm/codegen/axelera-codegen"
rm -rf "${_software_platform}"

# Pre-split layout: codegen still sits under host/lib, so the fallback picks it up.
_software_platform=$(make_sp host/tvm/axelera_compiler host/lib/codegen/axelera-codegen)
out=$(_install_alto_dev_shadow)
expect legacy-layout "$out" "INSTALL ${_software_platform}/host/lib/codegen/axelera-codegen"
rm -rf "${_software_platform}"

# Both layouts present, which is every machine that predates the move and then ran
# west update: the live ALTO checkout wins over the untracked host/lib leftover.
_software_platform=$(make_sp host/tvm/axelera_compiler host/tvm/codegen/axelera-codegen)
make_stale "${_software_platform}/host/lib/codegen/axelera-codegen"
out=$(_install_alto_dev_shadow)
expect both-present "$out" "INSTALL ${_software_platform}/host/tvm/codegen/axelera-codegen"
reject both-present "$out" "INSTALL ${_software_platform}/host/lib/codegen/axelera-codegen"
rm -rf "${_software_platform}"

# Leftover only, west update not run yet: the stale tree tracks nothing, so it is
# rejected rather than shadowed with dead code.
_software_platform=$(make_sp host/tvm/axelera_compiler)
make_stale "${_software_platform}/host/lib/codegen/axelera-codegen"
out=$(_install_alto_dev_shadow)
reject stale-only "$out" "INSTALL ${_software_platform}/host/lib/codegen/axelera-codegen"
expect stale-only "$out" "no axelera-codegen checkout found"
rm -rf "${_software_platform}"

# No source at all: warn and skip both, but still succeed (never abort the source).
_software_platform=$(make_sp)
out=$(_install_alto_dev_shadow) || { echo "FAIL: absent sources aborted the shadow" >&2; exit 1; }
expect absent "$out" "no axelera-compiler checkout found"
expect absent "$out" "no axelera-codegen checkout found"
reject absent "$out" "INSTALL"
rm -rf "${_software_platform}"

# Skew: a wheel built from another revision than the local checkout warns.
_software_platform=$(make_sp host/tvm/axelera_compiler host/tvm/codegen/axelera-codegen)
_installed_version="0.0.0.dev0+git.deadbeef"
out=$(_install_alto_dev_shadow)
expect skew "$out" "shadowing the axelera-compiler wheel (git.deadbeef)"

# Match: the wheel is that same revision, so no warning.
_installed_version="0.0.0.dev0+git.$(git -C "${_software_platform}/host/tvm/axelera_compiler" rev-parse --short=8 HEAD)"
out=$(_install_alto_dev_shadow)
reject skew-match "$out" "shadowing the axelera-compiler wheel"

# Release wheel: no git.<sha> to compare against, so no warning either.
_installed_version="1.8.0"
out=$(_install_alto_dev_shadow)
reject skew-release "$out" "shadowing the axelera-compiler wheel"
rm -rf "${_software_platform}"

# The three simulator pieces a developer used to paste by hand. Each is installed for them,
# only when it is actually absent, and only under --alto-dev.
eval "$(sed -n '/^_install_simulator_pieces() {/,/^}/p' "${script}")"

_sim_import_rc=1
_runtime_version="1.9.0a2.post19"
python() {
    case "$*" in
        *ax_accel_sim*) return "${_sim_import_rc}" ;;
        *importlib.metadata*) echo "${_runtime_version}" ;;
    esac
}
_pip_install() { printf 'INSTALL %s\n' "$*"; }
_install_dtc_into_venv() { printf 'INSTALL dtc\n'; }
_install_pinned_simulator() { printf 'INSTALL pinned-ax-accel-sim\n'; }
# The installed simulator's plugins resolve unless a case below says otherwise.
_plugins_resolve_rc=0
_simulator_plugins_resolve() { return "${_plugins_resolve_rc}"; }
# The runtime is new enough unless a case below says otherwise.
_version_at_least() { return "${_runtime_too_old:-0}"; }

# Nothing installed yet: all three pieces go in. The simulator comes from the pinned repaired
# wheel, never from `axelera-runtime[simulator]`, whose version-based resolve can only reach the
# wheel that does not run.
out=$(PATH=""; _install_simulator_pieces)
expect pieces-all "$out" "INSTALL pinned-ax-accel-sim"
expect pieces-all "$out" "INSTALL cmake"
expect pieces-all "$out" "INSTALL dtc"
reject pieces-all "$out" "axelera-runtime[simulator]"

# Installed is not the same as usable: the wheel imports and its plugins do not load, which is
# what the index serves at the revision's plain version. The repaired wheel replaces it.
_sim_import_rc=0
_plugins_resolve_rc=1
out=$(PATH=""; _install_simulator_pieces)
expect pieces-unrepaired "$out" "INSTALL pinned-ax-accel-sim"
_plugins_resolve_rc=0

# The simulator imports and its plugins load: only the other two are missing.
out=$(PATH=""; _install_simulator_pieces)
reject pieces-sim-present "$out" "pinned-ax-accel-sim"
expect pieces-sim-present "$out" "INSTALL cmake"

# cmake and dtc resolve (a shell function is what `command -v` finds): nothing to install.
cmake() { :; }
dtc() { :; }
out=$(PATH=""; _install_simulator_pieces)
reject pieces-none "$out" "INSTALL"

# A runtime older than the wheel-driven simulator is REPORTED, not silently accepted: it drives
# spike from a software-platform source tree and fails at the first kernel run.
_runtime_too_old=1
out=$(PATH=""; _install_simulator_pieces 2>&1)
expect pieces-old-runtime "$out" "1.9.0a2.post13 or newer"
_runtime_too_old=0
out=$(PATH=""; _install_simulator_pieces 2>&1)
reject pieces-new-runtime "$out" "1.9.0a2.post13 or newer"
unset -f cmake dtc

# Without the flag this does nothing at all, whatever is missing.
_sim_import_rc=1
_opt_alto_dev=false
out=$(PATH=""; _install_simulator_pieces)
reject pieces-flag-off "$out" "INSTALL"
_opt_alto_dev=true
unset -f python _pip_install _install_dtc_into_venv _simulator_plugins_resolve
unset -f _install_pinned_simulator _version_at_least

# The rootless dtc install itself: it unpacks the archive's own package into the venv, links the
# tools into its bin and needs root nowhere. Both apt-get and dpkg run unprivileged.
#
# OPT-IN, because this one case downloads a package from the distribution mirror. Every other
# check in this file is offline, and this file runs from the framework unit-test suite
# (tests/test_containerless_shell_tests_run.py), where a network fetch makes an unrelated
# change fail on a mirror outage. Set AXSCRIPT_NETWORK_TESTS=1 to run it; the kernel-lane
# setup doc names it beside the dtc step it covers. The reachability probe stays as a second
# condition, so an opted-in run with no mirror still skips instead of failing.
if [[ "${AXSCRIPT_NETWORK_TESTS:-0}" == "0" ]]; then
    echo "SKIP rootless-dtc: set AXSCRIPT_NETWORK_TESTS=1 (this case downloads a package)"
elif ! apt-get download --print-uris device-tree-compiler >/dev/null 2>&1; then
    echo "SKIP rootless-dtc: no apt archive reachable"
else
    eval "$(sed -n '/^_install_dtc_into_venv() {/,/^}/p' "${script}")"
    _lastline() { "$@"; }
    VIRTUAL_ENV=$(mktemp -d)
    mkdir -p "${VIRTUAL_ENV}/bin"
    _install_dtc_into_venv >/dev/null || { echo "FAIL: rootless dtc install failed" >&2; exit 1; }
    [[ -x "${VIRTUAL_ENV}/bin/dtc" ]] || { echo "FAIL: dtc did not land in the venv bin" >&2; exit 1; }
    PATH="${VIRTUAL_ENV}/bin:${PATH}" dtc --version >/dev/null \
        || { echo "FAIL: the installed dtc does not run" >&2; exit 1; }
    rm -rf "${VIRTUAL_ENV}"
    unset VIRTUAL_ENV
    unset -f _lastline
fi

# The option is not sticky, so an exported AXELERA_ALTO_DEV=1 is the only way the shadow
# can be on by default, and --no-alto-dev is what turns it back off.
eval "$(sed -n '/^_parse_args() {/,/^}/p' "${script}")"

alto_dev_after() {  # alto_dev_after <AXELERA_ALTO_DEV> [args...]
    local AXELERA_ALTO_DEV="$1"
    shift
    # The env default is an && list, so it returns 1 whenever the var is off.
    eval "$(sed -n '/^_opt_alto_dev=false$/,/^\[\[ "\${AXELERA_ALTO_DEV/p' "${script}")" || true
    _parse_args "$@" >/dev/null
    echo "${_opt_alto_dev}"
}

expect env-on "$(alto_dev_after 1)" true
expect env-on-flag-off "$(alto_dev_after 1 --no-alto-dev)" false
expect env-off-flag-on "$(alto_dev_after 0 --alto-dev)" true

echo "PASS"
