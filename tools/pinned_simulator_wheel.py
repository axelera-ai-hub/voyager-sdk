# Copyright Axelera AI, 2026
"""Print the ax-accel-sim version to install for the revision west.yml pins.

Two wheels are published per revision and only one of them runs. `auditwheel repair` vendors
the four libraries spike dlopens and applies a manylinux tag; without it the wheel imports
fine and dies minutes into a run, after the kernel has compiled, as `Failed to create executor
for model: ai_core_kernel_0`. The repaired wheel currently reaches the index under a version
with `.dirty` appended, because the job that runs cibuildwheel builds from a tree that west
update has touched, so pinning the honest version selects the raw one.

Selection is therefore by wheel FILENAME, never by version ordering: the newest file that
carries the pinned revision, this interpreter's tag, this architecture and a manylinux
platform. That keeps working once the publisher stops stamping `.dirty`, because the property
asked for is the one that matters.

    python tools/pinned_simulator_wheel.py                 # version, for `pip install ax-accel-sim==`
    python tools/pinned_simulator_wheel.py --revision SHA   # a revision other than west's pin
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import sysconfig
import urllib.request

DEFAULT_INDEX = (
    "https://artifactory.production.aws.axelera.ai/artifactory/api/pypi/swci-internal-pypi/simple"
)
PACKAGE = "ax-accel-sim"
_WHEEL = re.compile(r"ax_accel_sim-[^\"#]+\.whl")


def west_revision(west_yml: pathlib.Path, project: str = "riscv-isa-sim") -> str:
    """The revision west.yml pins `project` at.

    Read with a scan rather than a YAML parser: this runs from containerless.sh before any
    environment exists, so it cannot depend on pyyaml being installed.
    """
    seen = False
    for line in west_yml.read_text().splitlines():
        if re.match(rf"\s*-\s*name:\s*{re.escape(project)}\s*$", line):
            seen = True
        elif seen and (m := re.match(r"\s*revision:\s*([0-9a-fA-F]{7,40})\s*$", line)):
            return m.group(1)
        elif seen and re.match(r"\s*-\s*name:\s*", line):
            break
    raise SystemExit(f"no revision for west project {project!r} in {west_yml}")


def select(filenames: list[str], revision: str, py_tag: str, arch: str) -> str:
    """The version whose wheel is usable here, or "" when the revision published none.

    Every condition is a property the wheel must have to run, so a filename that satisfies all
    of them is the answer whatever its version string looks like.
    """
    short = revision[:8]
    usable = sorted(
        {
            name
            for name in filenames
            if f"git.{short}" in name and py_tag in name and "manylinux" in name and arch in name
        }
    )
    if not usable:
        return ""
    # One version per filename: ax_accel_sim-<version>-<py>-<abi>-<platform>.whl
    return usable[-1].split("-")[1]


def find_west_yml(start: pathlib.Path) -> pathlib.Path | None:
    """The enclosing software-platform's west.yml, or None.

    Walked rather than computed from a fixed depth: a git worktree of this repository lives
    outside the software-platform tree, so `parents[3]` names a path that does not exist.
    """
    for parent in [start, *start.parents]:
        if (parent / "west.yml").is_file():
            return parent / "west.yml"
    return None


def main(argv: list[str] | None = None) -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", help="git revision to match (default: west.yml's pin)")
    p.add_argument(
        "--west-yml",
        type=pathlib.Path,
        default=find_west_yml(repo_root),
        help="path to software-platform's west.yml (default: the enclosing checkout's)",
    )
    p.add_argument("--index-url", default=DEFAULT_INDEX)
    args = p.parse_args(argv)

    if not args.revision and args.west_yml is None:
        p.error(
            "no west.yml above this checkout (a git worktree lives outside the "
            "software-platform tree); pass --west-yml or --revision"
        )
    revision = args.revision or west_revision(args.west_yml)
    py_tag = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    arch = sysconfig.get_platform().rsplit("-", 1)[-1]
    with urllib.request.urlopen(f"{args.index_url}/{PACKAGE}/", timeout=30) as resp:
        page = resp.read().decode("utf-8", "replace")
    version = select(_WHEEL.findall(page), revision, py_tag, arch)
    if not version:
        print(
            f"no repaired {PACKAGE} wheel for revision {revision[:8]} ({py_tag}, {arch}) on "
            f"{args.index_url}; build one with tools/build_simulator_wheel.sh",
            file=sys.stderr,
        )
        return 1
    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
