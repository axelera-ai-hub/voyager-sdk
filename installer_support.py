#!/usr/bin/env python3
# Copyright Axelera AI, 2024
import re
import subprocess
import sys

"""
This module provides support functions for the installer scripts.
It is very much WIP and is expected to grow and change considerably as
existing bash code in the installer scripts is ported to python, and extended.
"""


def run(cmd, shell=True, check=True, verbose=False, capture_output=True):
    if verbose:
        print(cmd)
    try:
        result = subprocess.run(
            cmd, shell=shell, check=check, capture_output=capture_output, text=True
        )
        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        raise


def check_depends(cmd):
    """Takes a failed install command and reruns it, examining the output for dependency issues.
    It collects the names of the interrelated packages and prints them to stdout so the results
    can be used in bash code to correct the failed install."""
    matches = []
    while True:
        try:
            run(cmd)
        except subprocess.CalledProcessError as e:
            output = e.stdout + e.stderr
            new_matches = []
            for line in output.splitlines():
                match = re.match(r"^\s*([^\s]*).*(?:Depends|Breaks): ([^\s]*)", line)
                if match:
                    for g in match.groups():
                        new_matches.append(g)
            if new_matches:
                matches.extend(new_matches)
                cmd += f" {' '.join(new_matches)}"
            else:
                break
    print(" ".join(matches))
