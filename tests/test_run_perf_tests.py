# Copyright Axelera AI, 2026
from pathlib import Path
import re

import pytest

import run_perf_tests


BASELINE = {
    "software-platform": "e2fedc81eede652c39c5164b689925599b149d74",
    "application.framework": "800ce8642db587bb227f6eab51a0fa50cf8c96cb",
    "prebuilt-version": "1.8.0-rc1",
}
CANDIDATE = {
    "software-platform": "7d4f6f856b68405999923a9e34efc6e88e8f3630",
    "application.framework": "2222222222222222222222222222222222222222",
    "prebuilt-version": "1.8.0-rc3",
}


def test_format_provenance_report_distinguishes_baseline_and_candidate():
    # Given two independently reproducible source environments
    baseline = BASELINE
    candidate = CANDIDATE

    # When the report provenance is rendered
    report = run_perf_tests._format_provenance_report(baseline, candidate)

    # Then each row contains its own immutable SP and AF revisions
    assert "| Baseline | `e2fedc81eede652c39c5164b689925599b149d74` |" in report
    assert "`800ce8642db587bb227f6eab51a0fa50cf8c96cb` |" in report
    assert "| Candidate | `7d4f6f856b68405999923a9e34efc6e88e8f3630` |" in report
    assert "`2222222222222222222222222222222222222222` |" in report
    assert "`1.8.0-rc1` |" in report
    assert "`1.8.0-rc3` |" in report


def test_current_provenance_reads_sp_and_af_checkouts(monkeypatch, tmp_path):
    # Given the software-platform workspace used by the CI container
    workspace = tmp_path / "software-platform"
    framework = workspace / "host" / "application" / "framework"
    framework.mkdir(parents=True)
    revisions = {framework: CANDIDATE["application.framework"]}
    visited = []

    def fake_git_revision(path: Path, revision: str = "HEAD") -> str:
        visited.append(path)
        assert revision == "HEAD^{commit}"
        return revisions[path]

    monkeypatch.setattr(run_perf_tests, "_git_revision", fake_git_revision)
    monkeypatch.setattr(
        run_perf_tests,
        "_installed_software_platform_revision",
        lambda path, ci_mode: CANDIDATE["software-platform"],
    )

    # When current provenance is collected
    provenance = run_perf_tests._current_provenance(workspace, CANDIDATE["prebuilt-version"], True)

    # Then it identifies both repositories independently
    assert provenance == CANDIDATE
    assert visited == [framework]


def test_installed_software_platform_revision_uses_axelera_sp_ref(monkeypatch, tmp_path):
    # Given a local/containerless software-platform reference
    workspace = tmp_path / "software-platform"
    monkeypatch.setenv("AXELERA_SP_REF", "7d4f6f856")

    def fake_git_revision(path: Path, revision: str = "HEAD") -> str:
        assert path == workspace
        assert revision == "7d4f6f856^{commit}"
        return CANDIDATE["software-platform"]

    monkeypatch.setattr(run_perf_tests, "_git_revision", fake_git_revision)

    # When the installed software-platform revision is resolved in local mode
    revision = run_perf_tests._installed_software_platform_revision(workspace, ci_mode=False)

    # Then the configured reference is expanded by the checkout
    assert revision == CANDIDATE["software-platform"]


def test_main_reports_provenance_when_measurements_fail(monkeypatch, tmp_path, capsys):
    # Given a reproducible environment whose measurement command fails
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path / "software-platform"))
    monkeypatch.setattr(run_perf_tests, "_load_provenance", lambda path: BASELINE)
    monkeypatch.setattr(
        run_perf_tests,
        "_current_provenance",
        lambda workspace, prebuilt_version, ci_mode, application_framework_revision: CANDIDATE,
    )
    monkeypatch.setattr(
        run_perf_tests.test_helpers,
        "run",
        lambda command, **kwargs: (_ for _ in ()).throw(RuntimeError("download failed")),
    )

    # When the CI entry point runs
    with pytest.raises(RuntimeError, match="download failed"):
        run_perf_tests.main()

    # Then it preserves the failure while emitting actionable environment details
    output = capsys.readouterr().out
    assert run_perf_tests.REPORT_START_MARKER in output
    assert run_perf_tests.REPORT_END_MARKER in output
    assert CANDIDATE["software-platform"] in output
    assert CANDIDATE["application.framework"] in output
    assert ":warning: Performance measurements were not produced." in output


def test_committed_golden_provenance_identifies_a_reproducible_ci_run():
    # Given the provenance committed beside the golden measurements
    provenance_path = Path(run_perf_tests.__file__).with_name("golden_perf_results.json")

    # When the provenance is loaded
    provenance = run_perf_tests._load_provenance(provenance_path)

    # Then it identifies the run that produced them: a resolved commit per repository, and
    # the prebuilt they were measured against. Deliberately NOT compared against BASELINE:
    # `update-golden-perf.yml` rewrites this file whenever the measurements move, and a
    # fixture in a test file is not something that automation can know to update. Pinning
    # one exact pair here made every PR red the next time the golden was refreshed, while
    # saying nothing about whether the file is usable.
    assert set(provenance) == {*run_perf_tests.PROVENANCE_REPOSITORIES, "prebuilt-version"}
    for repository in run_perf_tests.PROVENANCE_REPOSITORIES:
        assert re.fullmatch(r"[0-9a-f]{40}", provenance[repository]), (
            f"{repository} provenance {provenance[repository]!r} is not a resolved commit; "
            f"a report cannot point a reader at the source that produced these numbers"
        )
    # `_load_provenance` substitutes "unknown" for an absent key, so this is the one value
    # that means the field was never written rather than written wrongly.
    assert provenance["prebuilt-version"] not in ("", "unknown")
