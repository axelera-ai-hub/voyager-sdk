# Copyright Axelera AI, 2026
"""Every pull request reaches `precommit-global`, and only a pull request can.

`framework-ci` used to run `pre-commit --all-files`, so pre-commit covered a pull
request that no path filter matched. Now that only `precommit-global` runs it,
over the changed files, two combinations would silently get no pre-commit at all:
a run of an event that carries no base commit to diff against, and a pull request
whose every file a docs filter excluded. Neither turns a job red, so both are
asserted here rather than reviewed.

The workflow itself carries a scheduled trigger for the AxScript kernel lane, so
"only a pull request" is asserted of the JOB rather than of the trigger list: every
job that reads `github.event.pull_request` states `github.event_name ==
'pull_request'`, and a path or `needs` gate on `precommit-global` stays forbidden.
"""

import pathlib
import re

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / '.github/workflows/python.yml'
PRECOMMIT_CONFIG = REPO_ROOT / '.pre-commit-config.yaml'

A_MARKDOWN_FILE = 'axelera_zoo/docs/internal/recipes.md'


def _workflow():
    parsed = yaml.safe_load(WORKFLOW.read_text())
    assert parsed['jobs'], f'{WORKFLOW} parsed to no jobs'
    return parsed


def _hooks_that_read(path):
    """The hooks pre-commit would run for `path`, by the config alone.

    Only hooks whose selection this file fully determines: a `files` or `types`
    restriction is pre-commit's to evaluate, so those are left out rather than
    guessed at.
    """
    config = yaml.safe_load(PRECOMMIT_CONFIG.read_text())
    globally_excluded = re.search(config['exclude'], path, re.X)
    reading = set()
    for repo in config['repos']:
        for hook in repo['hooks']:
            if globally_excluded or hook.get('files') or hook.get('types'):
                continue
            if hook.get('exclude') and re.search(hook['exclude'], path):
                continue
            reading.add(hook['id'])
    return reading


#: Jobs that diff against `github.event.pull_request.base.sha`, or otherwise read the
#: pull request. On any other event they check nothing, and `precommit-global` PASSES
#: having done so, which is the failure this file exists for. `changes` is here because
#: paths-filter falls back to the last commit on the default branch when there is no pull
#: request, so its outputs would describe MAIN'S TIP and every job gating on them would run.
PULL_REQUEST_ONLY_JOBS = (
    'changes',
    'precommit-global',
    'copyright-check',
    'markdown-link-checker',
)


def test_the_workflow_still_runs_on_every_pull_request():
    parsed = _workflow()
    # `on` is the YAML 1.1 boolean true.
    triggers = parsed[True] if True in parsed else parsed['on']
    keys = triggers if isinstance(triggers, (list, tuple)) else list(triggers)
    assert 'pull_request' in keys, f'python.yml must trigger on pull_request; it has {keys}'


def test_only_a_pull_request_reaches_the_jobs_that_read_one():
    """The workflow also runs on a schedule, for the AxScript kernel lane.

    `precommit-global` diffs against `github.event.pull_request.base.sha`, which no
    other event carries: unguarded, the step resolves the base to empty, reports no
    files to check, and passes. Asserted per job, because the trigger list is no
    longer the thing that keeps it off a scheduled run.
    """
    jobs = _workflow()['jobs']
    for name in PULL_REQUEST_ONLY_JOBS:
        assert jobs[name].get('if') == "github.event_name == 'pull_request'", (
            f'{name} reads the pull request, so it must state '
            f"if: github.event_name == 'pull_request'; it states {jobs[name].get('if')!r}"
        )


def test_every_pull_request_reaches_the_global_precommit():
    """A gate here is what let a Markdown-only pull request run no hook at all."""
    job = _workflow()['jobs']['precommit-global']
    assert 'needs' not in job, (
        'precommit-global must not wait on the `changes` job: a path filter is what let a '
        'Markdown-only pull request run no hook at all'
    )
    # The event guard above is the ONLY condition allowed here. Anything else is a gate on
    # which files changed, wearing an `if:`.
    assert job.get('if') == "github.event_name == 'pull_request'", (
        'precommit-global must be gated on the event alone so it covers every pull request, '
        f'including one that changes only Markdown; it states {job.get("if")!r}'
    )


def test_a_markdown_only_pull_request_has_hooks_left_to_run():
    """Proves the job above is not a no-op on the pull request that motivated it."""
    reading = _hooks_that_read(A_MARKDOWN_FILE)
    assert {'end-of-file-fixer', 'mixed-line-ending'} <= reading, (
        f'{A_MARKDOWN_FILE} should still reach the line-ending hooks, but only '
        f'{sorted(reading)} read it'
    )
    # The one that does skip Markdown, so a reader does not conclude from the
    # names above that every whitespace hook covers it.
    assert 'trailing-whitespace' not in reading


def test_the_guard_actually_covers_something():
    assert WORKFLOW.is_file() and PRECOMMIT_CONFIG.is_file()
    assert 'precommit-global' in _workflow()['jobs']
    # A path the global exclude swallows must come back empty, or the helper above
    # is reporting hooks it never actually filtered.
    assert _hooks_that_read('operators/PillowResize/PillowResize.cpp') == set()
