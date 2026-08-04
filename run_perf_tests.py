#!/usr/bin/env python
# Copyright Axelera AI, 2026
import subprocess


def main():
    subprocess.run(['./run_perf_tests.sh'], check=True)


if __name__ == '__main__':
    main()
