#!/usr/bin/env python
# Copyright Axelera AI, 2024
"""Simple tool to create wheels for the unittests."""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import tempfile

# note paths relative to software-platform
WHEELS = [
    ('axelera_runtime', 'host/lib/axelera-runtime'),
    ('axelera_types', 'host/lib/types'),
]


def _run(args, cwd):
    opts = dict(capture_output=True, check=True, text=True)
    return subprocess.run(args, cwd=cwd, **opts).stdout


def find_software_platform():
    here = Path(__file__).parent
    while here.name != "software-platform":
        here = here.parent
        if here == Path("/"):
            raise RuntimeError("Could not find software-platform")
    return here


def build_wheel(path: Path) -> Path:
    stdout = _run(["pip", "wheel", "."], cwd=path)
    for line in stdout.splitlines():
        if m := re.match(r"\s*Created wheel for [^:]+: filename=(\S+)", line):
            return path / m.group(1)
    raise RuntimeError(f"Could not find wheel file name in {stdout}")


def reversion_wheel(whl, name, dest, version):
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        stdout = _run(['wheel', 'unpack', str(whl)], cwd=td)
        unpacked = td / re.search(r"Unpacking to: (.+)...OK", stdout).group(1)
        dist_info = unpacked / f"{unpacked.name}.dist-info"
        meta = (dist_info / "METADATA").read_text()
        meta = re.sub(r"^Version: .+$", f"Version: {version}", meta, flags=re.MULTILINE)
        (dist_info / "METADATA").write_text(meta)
        dist_info.rename(unpacked / f"{name}-{version}.dist-info")
        unpacked.rename(td / f"{name}-{version}")
        _run(['wheel', 'pack', f"{name}-{version}"], cwd=td)

        wheel_files = list(td.glob(f"{name}-{version}*.whl"))
        if not wheel_files:
            raise RuntimeError(f"Could not find packed wheel for {name}")
        newwhl = wheel_files[0]

        if dest.exists():
            dest.unlink()
        shutil.copy2(newwhl, dest)


def main():
    sp = find_software_platform()
    af = sp / "host/application/framework"
    version = '9.9.9'
    wheels_dir = af / "wheels_for_tests"
    wheels_dir.mkdir(exist_ok=True)

    for name, path in WHEELS:
        whl = build_wheel(sp / path)
        print(whl)
        dest = wheels_dir / f"{name}-{version}-py3-none-any.whl"
        reversion_wheel(whl, name, dest, version)
        print(whl.relative_to(sp), "->", dest.relative_to(af))


if __name__ == "__main__":
    main()
