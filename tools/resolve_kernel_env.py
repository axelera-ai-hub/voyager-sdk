# Copyright Axelera AI, 2026
"""Print the AxScript kernel environment's requirement lines for one software-platform revision.

The kernel source under `model_axscript/llm/` is written against a compiler SOURCE API, so a
compiler and a framework tree either match or nothing compiles. This repository therefore states
NO component version of its own. It names one software-platform revision, and everything else is
read from what software-platform published for it:

    axelera-devkit, axelera-rt   the wheels built from that software-platform revision
    axelera-compiler, -codegen,  the wheels built from the `host.compiler.alto` revision that
    -tvm                         revision's west.yml pins
    ax-accel-sim                 the wheel built from the `riscv-isa-sim` revision it pins

    python tools/resolve_kernel_env.py --part devkit
    python tools/resolve_kernel_env.py --part compiler --sp-revision main
    python tools/resolve_kernel_env.py --print-revisions

Without `--sp-revision` it uses the enclosing software-platform checkout's HEAD. With one it
fetches west.yml from the software-platform repository, for a CI job that clones this repository
alone, and needs a token in GITHUB_TOKEN because that repository is private.

Three parts rather than one list, because they are three pip resolves. `axelera-devkit` depends
on the compiler at its own bare release label, so asking for the devkit and a post-release
compiler in one resolve is `ResolutionImpossible`; the compiler part is installed second and
overrides what the devkit brought.

Selection is by wheel FILENAME, never by version ordering, because a revision is what is being
asked for and the version string carrying it is not predictable: the devkit, compiler and
codegen spell it `+git.<8 hex>` while axelera-tvm spells it `+g<9 hex>`.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_INDEX = (
    "https://artifactory.production.aws.axelera.ai/artifactory/api/pypi/swci-internal-pypi/simple"
)
#: Versioned by the software-platform revision itself. The devkit extra is `hf-transformers` and
#: deliberately not `test`: `test` pulls axelera-evalsuite, then ultralytics, which installs a
#: REGULAR top-level package named `tests` that shadows this repository's namespace `tests/`
#: directory and stops the imported kernel suite importing. Nothing here uses evalsuite.
SP_PACKAGES = ("axelera-devkit[hf-transformers]", "axelera-rt")
#: Versioned by the revision west.yml pins for this project.
ALTO_PROJECT = "host.compiler.alto"
ALTO_PACKAGES = ("axelera-compiler", "axelera-codegen", "axelera-tvm")
#: The simulator's own west project. Its wheel needs more than a revision match; see
#: pinned_simulator_wheel.select.
SIM_PROJECT = "riscv-isa-sim"
#: Not resolvable from a revision, so stated: a run-time dependency of a kernel compile that
#: nothing in the wheel dependency graph declares, plus the test runner itself.
PLAIN = ("pytest", "pytest-timeout", "pytest-xdist", "cmake")
_WEST_YML_URL = "https://raw.githubusercontent.com/axelera-ai/software-platform/{rev}/west.yml"
_COMMIT_URL = "https://api.github.com/repos/axelera-ai/software-platform/commits/{rev}"
_SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _read_url(url: str, token: str | None = None, timeout: int = 90) -> str:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(request, timeout=timeout).read().decode()


def software_platform_root(start: pathlib.Path) -> pathlib.Path | None:
    """The enclosing software-platform checkout, or None.

    Walked rather than computed from a fixed depth: a git worktree of this repository lives
    outside the software-platform tree, so a parent count names a path that does not exist.
    """
    for parent in [start, *start.parents]:
        if (parent / "west.yml").is_file():
            return parent
    return None


def west_yml_and_revision(sp_revision: str | None, token: str | None) -> tuple[str, str]:
    """(west.yml text, the software-platform revision it came from).

    Raises rather than falling back to any other copy. Reading a stale west.yml is the failure
    this script exists to remove: it names a compiler that cannot compile the tree, and the
    error that follows points at a kernel rather than at the pin.
    """
    if sp_revision is None:
        root = software_platform_root(pathlib.Path.cwd().resolve())
        if root is None:
            raise SystemExit(
                "no enclosing software-platform checkout. Pass --sp-revision <sha|main> to "
                "read west.yml from the software-platform repository instead."
            )
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if head.returncode != 0:
            raise SystemExit(f"{root} has a west.yml but is not a git checkout")
        return (root / "west.yml").read_text(), head.stdout.strip()
    hint = "" if token else " (software-platform is private: set GITHUB_TOKEN)"
    if not _SHA.match(sp_revision):
        # A branch name has to become a commit before anything else happens: the devkit's
        # version carries the software-platform SHA, so matching wheel filenames against the
        # string "main" finds nothing and reads as an unpublished revision.
        try:
            body = _read_url(_COMMIT_URL.format(rev=sp_revision), token, timeout=60)
        except urllib.error.HTTPError as exc:
            raise SystemExit(
                f"cannot resolve software-platform {sp_revision!r} to a commit: "
                f"HTTP {exc.code}{hint}"
            ) from exc
        m = re.search(r'"sha"\s*:\s*"([0-9a-f]{40})"', body)
        if m is None:
            raise SystemExit(f"no commit sha in the API response for {sp_revision!r}")
        sp_revision = m.group(1)
    try:
        return _read_url(_WEST_YML_URL.format(rev=sp_revision), token), sp_revision
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"cannot read west.yml at {sp_revision!r}: HTTP {exc.code}{hint}"
        ) from exc


def west_revision(west_yml_text: str, project: str) -> str:
    """The revision `west.yml` pins `project` at.

    Scanned rather than parsed, so this runs before any environment exists and cannot depend on
    pyyaml. `pinned_simulator_wheel.west_revision` reads the same file from a path; this takes
    text, because a CI job fetches it and has no file to hand over.
    """
    seen = False
    for line in west_yml_text.splitlines():
        if re.match(rf"\s*-\s*name:\s*{re.escape(project)}\s*$", line):
            seen = True
        elif seen and (m := re.match(r"\s*revision:\s*([0-9a-fA-F]{7,40})\s*$", line)):
            return m.group(1)
        elif seen and re.match(r"\s*-\s*name:\s*", line):
            break
    raise SystemExit(f"no revision for west project {project!r} in west.yml")


def index_filenames(package: str, index: str) -> set[str]:
    base = urllib.parse.urljoin(index.rstrip("/") + "/", f"{package}/")
    try:
        body = _read_url(base)
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot read the index for {package}: {exc}") from exc
    return set(re.findall(rf"{package.replace('-', '_')}-[^\"#]+\.whl", body))


def version_for(package: str, revision: str, index: str) -> str | None:
    """The newest published version of `package` whose wheel carries `revision`, or None.

    Both local-version spellings are matched, because the packages disagree: `+git.<8 hex>` and
    `+g<9 hex>`.
    """
    short = revision[:8]
    hits = {
        name
        for name in index_filenames(package, index)
        if f"git.{short}" in name or f"+g{short}" in name
    }
    # One version per filename: <stem>-<version>-<py>-<abi>-<platform>.whl
    return sorted({n.split("-")[1] for n in hits})[-1] if hits else None


def _pin_all(packages, revision: str, index: str, what: str) -> list[str]:
    """`name==version` for each package, or exit naming every one that was not published.

    Exits rather than dropping a package. Installing two of three leaves a compiler and a
    codegen that disagree, which fails deep inside a compile with a message about the kernel.
    """
    pinned, missing = [], []
    for package in packages:
        name = package.partition("[")[0]
        version = version_for(name, revision, index)
        if version is None:
            missing.append(name)
        else:
            pinned.append(f"{package}=={version}")
    if missing:
        raise SystemExit(
            f"no published wheel for {', '.join(missing)} at {what} revision {revision[:8]}. "
            "Either the publish job has not finished for that revision or it failed. Check "
            "the index rather than pinning a different revision by hand."
        )
    return pinned


def simulator_requirement(revision: str, index: str) -> str:
    """`ax-accel-sim==<version>` for the runnable wheel at `revision`.

    Delegated to pinned_simulator_wheel, because two wheels are published per revision and only
    the auditwheel-repaired one runs: the other imports fine and then dies minutes into a run,
    after the kernel has compiled, as "Failed to create executor for model: ai_core_kernel_0".
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import sysconfig

    from pinned_simulator_wheel import PACKAGE, select

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    arch = (sysconfig.get_platform().split("-")[-1] or "x86_64").replace("universal2", "x86_64")
    version = select(sorted(index_filenames(PACKAGE, index)), revision, py_tag, arch)
    if not version:
        raise SystemExit(
            f"no runnable {PACKAGE} wheel for riscv-isa-sim revision {revision[:8]} at "
            f"{py_tag}/{arch}. A wheel that is published but not auditwheel-repaired does not "
            "count; see tools/pinned_simulator_wheel.py."
        )
    return f"{PACKAGE}=={version}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--part",
        choices=("devkit", "compiler", "simulator"),
        help="which pip resolve's requirement lines to print",
    )
    p.add_argument(
        "--sp-revision",
        help="software-platform revision (a sha, or 'main'). Default: the enclosing "
        "software-platform checkout's HEAD.",
    )
    p.add_argument("--index-url", default=DEFAULT_INDEX)
    p.add_argument(
        "--print-revisions",
        action="store_true",
        help="print the revisions this would resolve from, and install nothing",
    )
    args = p.parse_args(argv)
    if not args.part and not args.print_revisions:
        p.error("pass --part or --print-revisions")

    text, sp_revision = west_yml_and_revision(args.sp_revision, os.environ.get("GITHUB_TOKEN"))
    alto = west_revision(text, ALTO_PROJECT)
    sim = west_revision(text, SIM_PROJECT)

    if args.print_revisions:
        print(f"software-platform  {sp_revision}")
        print(f"{ALTO_PROJECT:18s} {alto}")
        print(f"{SIM_PROJECT:18s} {sim}")
        return 0

    if args.part == "devkit":
        lines = _pin_all(SP_PACKAGES, sp_revision, args.index_url, "software-platform")
        lines += list(PLAIN)
    elif args.part == "compiler":
        lines = _pin_all(ALTO_PACKAGES, alto, args.index_url, ALTO_PROJECT)
    else:
        lines = [simulator_requirement(sim, args.index_url)]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
