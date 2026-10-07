#!/usr/bin/env python
# Copyright Axelera AI, 2026
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Mapping

import test_helpers

root_path = Path(__file__).resolve().parent
golden_perf_results_path = root_path / "golden_perf_results.csv"
golden_perf_provenance_path = root_path / "golden_perf_results.json"


# Sentinel markers so the CI workflow can reliably extract the perf report from
# the (potentially noisy) container stdout and post it as a comment on the PR.
# Keep these in sync with .github/workflows/application-framework-perfrun.yml in
# the software-platform repo.
REPORT_START_MARKER = "<<<PERF_REPORT_MARKDOWN_START>>>"
REPORT_END_MARKER = "<<<PERF_REPORT_MARKDOWN_END>>>"
GOLDEN_FILES_START_MARKER = "<<<GOLDEN_PERF_FILES_JSON_START>>>"
GOLDEN_FILES_END_MARKER = "<<<GOLDEN_PERF_FILES_JSON_END>>>"

RESULTS_DIR_NAME = "run_perf_tests_results"

# The columns requested from `iax-apps-perf-report report`. The markdown header
# is derived from these field names, so the two can never drift apart even if
# format specs (e.g. `:.2f`) are added.
REPORT_FORMAT = "{name},{quickaccu_cpu:.2f},{quickaccu_cl:.2f}"

PROVENANCE_REPOSITORIES = ("software-platform", "application.framework")

# Columns kept from the raw quickaccu table when refreshing the golden CSV, matching
# the header golden_perf_results.csv already uses.
GOLDEN_TABLE_COLUMNS = ("name", "quickaccu(cpu)", "quickaccu(cl)")


def _git_revision(path: Path, revision: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", revision],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _installed_software_platform_revision(workspace: Path, ci_mode: bool) -> str:
    if ci_mode:
        return _git_revision(workspace)

    ref = os.environ.get("AXELERA_SP_REF")
    if not ref:
        raise RuntimeError(
            "AXELERA_SP_REF is not set. Activate a containerless environment first "
            "(it exports software_platform_ref as AXELERA_SP_REF)."
        )
    return _git_revision(workspace, f"{ref}^{{commit}}")


def _current_provenance(
    workspace: Path,
    prebuilt_version: str,
    ci_mode: bool,
    application_framework_revision: str = "HEAD",
) -> dict[str, str]:
    framework = workspace / "host" / "application" / "framework"
    return {
        "software-platform": _installed_software_platform_revision(workspace, ci_mode),
        "application.framework": _git_revision(
            framework, f"{application_framework_revision}^{{commit}}"
        ),
        "prebuilt-version": prebuilt_version,
    }


def _load_provenance(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    provenance = {repository: data[repository] for repository in PROVENANCE_REPOSITORIES}
    provenance["prebuilt-version"] = data.get("prebuilt-version", "unknown")
    return provenance


def _update_golden_files(table_path: str, provenance: Mapping[str, str]) -> None:
    # Imported lazily: only needed for --update-golden, which runs where
    # iax-apps-perf-report (and therefore pandas) is already installed.
    import pandas as pd

    table = pd.read_csv(table_path)
    table[list(GOLDEN_TABLE_COLUMNS)].to_csv(
        golden_perf_results_path, index=False, float_format="%.2f"
    )
    golden_perf_provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")


def _golden_files_payload() -> str:
    return json.dumps(
        {
            golden_perf_results_path.name: golden_perf_results_path.read_text(),
            golden_perf_provenance_path.name: golden_perf_provenance_path.read_text(),
        }
    )


def _format_provenance_report(baseline: Mapping[str, str], candidate: Mapping[str, str]) -> str:
    return "\n".join(
        (
            "#### Source revisions",
            "",
            "| Run | software-platform | application.framework | prebuilt version |",
            "| --- | --- | --- | --- |",
            f"| Baseline | `{baseline['software-platform']}` | "
            f"`{baseline['application.framework']}` | `{baseline['prebuilt-version']}` |",
            f"| Candidate | `{candidate['software-platform']}` | "
            f"`{candidate['application.framework']}` | `{candidate['prebuilt-version']}` |",
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version',
        default='1.9.0a3',
        help='Prebuilt version to download from (default: %(default)s)',
    )
    parser.add_argument('--CI', action='store_true', help='Run in CI mode')
    parser.add_argument(
        '--update-golden',
        action='store_true',
        help=(
            'Overwrite golden_perf_results.csv/json with the results of this run '
            'instead of diffing against them.'
        ),
    )
    parser.add_argument(
        '--AF-revision',
        dest='application_framework_revision',
        default='HEAD',
        help='Application framework revision to record in golden_perf_results.json.',
    )
    args, _ = parser.parse_known_args()

    # ensure we get the subset from s3 so inference can run
    os.environ["AXELERA_S3_AVAILABLE"] = "subset"

    # Start from a clean results directory to hold the intermediate CSV.
    if os.path.exists(RESULTS_DIR_NAME):
        shutil.rmtree(RESULTS_DIR_NAME)
    os.makedirs(RESULTS_DIR_NAME)
    table_path = os.path.join(RESULTS_DIR_NAME, "results.csv")
    baseline_provenance = _load_provenance(golden_perf_provenance_path)
    ci_mode = args.CI or os.environ.get("CI", "").lower() == "true"
    workspace = (
        Path(os.environ["GITHUB_WORKSPACE"])
        if "GITHUB_WORKSPACE" in os.environ
        else root_path.parents[2]
    )
    candidate_provenance = _current_provenance(
        workspace, args.version, ci_mode, args.application_framework_revision
    )
    report = None
    golden_updated = False

    try:
        requirements_ci = test_helpers.requirements_ci_path(root_path)
        test_helpers.run(f"pip install -r {requirements_ci}", verbose=3)
        test_helpers.run("pip install -e internal-apps", verbose=3)
        test_helpers.run(
            f'iax-apps-perf-report --minimal download --version {args.version}', verbose=3
        )
        test_helpers.run(
            f"iax-apps-perf-report --minimal --table-path {table_path} quickaccu",
            verbose=3,
        )
        if args.update_golden:
            _update_golden_files(table_path, candidate_provenance)
            golden_updated = True
        else:
            result, _ = test_helpers.run(
                f"iax-apps-perf-report --minimal --table-path {table_path} report "
                f'--format "{REPORT_FORMAT}" --golden {golden_perf_results_path} '
                "--report-format github-html",
                verbose=1,
            )
            report = result.stdout
    finally:
        print(REPORT_START_MARKER)
        print("### Application Framework Performance Report")
        print()
        print(_format_provenance_report(baseline_provenance, candidate_provenance))
        print()
        if golden_updated:
            print(":white_check_mark: golden_perf_results.csv/json updated from this run.")
        elif report is None:
            print(":warning: Performance measurements were not produced.")
        else:
            print(report)
        print(REPORT_END_MARKER)
        if golden_updated:
            print(GOLDEN_FILES_START_MARKER)
            print(_golden_files_payload())
            print(GOLDEN_FILES_END_MARKER)
        shutil.rmtree(RESULTS_DIR_NAME)


if __name__ == "__main__":
    main()
