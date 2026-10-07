#!/bin/bash
# Copyright Axelera AI, 2026
set -euo pipefail

root=$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")
fake_venv=$(mktemp -d)
trap 'rm -rf "${fake_venv}"' EXIT
info_file="${fake_venv}/.containerless-info"

printf '%s\n' \
    'containerless_env=all' \
    'containerless_hf=true' \
    'containerless_python=python3.10' \
    'software_platform_ref=ae63fd17113534409537f9e0d36720a009968c08' \
    'software_platform_dirty=false' \
    'application_framework_commit=d9c662413d524a89f7a3d90af0f353ab4e19a0cb' \
    'application_framework_dirty=true' > "${info_file}"

output=$(VIRTUAL_ENV="${fake_venv}" "${root}/containerless-info")
[[ "${output}" == $'software-platform: ae63fd17113534409537f9e0d36720a009968c08\napplication.framework: d9c662413d524a89f7a3d90af0f353ab4e19a0cb (dirty)\nrecreate from an application.framework checkout:\n  git checkout d9c662413d524a89f7a3d90af0f353ab4e19a0cb\n  . containerless.sh all ae63fd17113534409537f9e0d36720a009968c08 --hf --python \"$(command -v python3.10)\" --update\nwarning: dirty changes are not included in these commands' ]]

if env -u VIRTUAL_ENV "${root}/containerless-info" >/dev/null 2>&1; then
    echo "containerless-info unexpectedly succeeded without an active venv" >&2
    exit 1
fi

rm "${info_file}"
if VIRTUAL_ENV="${fake_venv}" "${root}/containerless-info" >/dev/null 2>&1; then
    echo "containerless-info unexpectedly succeeded without provenance" >&2
    exit 1
fi
