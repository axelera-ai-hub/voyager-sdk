#!/usr/bin/env python
# Copyright Axelera AI, 2026
import time
import subprocess
import re
import sys
from pathlib import Path

_RE_ERROR = re.compile(r'(^ERROR\s)|(Traceback \(most recent call last\))')


def requirements_ci_path(root_path: Path) -> str:
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    return str(root_path / "requirements" / f"py{py}" / "requirements.CI.txt")


def _format_stdouterr(result, stdname, context):
    if s := getattr(result, stdname, ''):
        lines = s.splitlines()
        if context and len(lines) > context * 2 + 1:
            lines = lines[:context] + ['...'] + lines[-context:]
        lines = [f'{stdname}: '] + lines
    else:
        lines = [f'{stdname}: <none>']
    return '\n'.join(lines)


def run(cmd, shell=True, verbose=None, timing=False, dry_run=False):
    verbose = verbose or 0
    if verbose or dry_run:
        print(cmd)
    if dry_run:
        return subprocess.CompletedProcess(cmd, returncode=0), None
    if timing:
        start = time.time()
    result = subprocess.run(cmd, shell=shell, check=False, capture_output=True, text=True)
    outerr = result.stdout + result.stderr
    if result.returncode > 0 or _RE_ERROR.search(outerr):
        # raise here instead of using check=True because pytest insists on showing traceback from
        # subprocess even if we use `raise e from None`
        if result.returncode == 0:
            msg = f'{cmd}\nappears to have failed but has returned no error code'
        else:
            msg = f"{cmd} failed with returncode={result.returncode}\n"
        context = [3, 10, 30, None][min(verbose, 3)]
        raise RuntimeError(
            msg + f"{_format_stdouterr(result, 'stdout', context)}\n"
            f"{_format_stdouterr(result, 'stderr', context)}"
        )
    if verbose > 1 and outerr:
        print(outerr)
    return result, (time.time() - start if timing else None)
