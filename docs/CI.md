# CI contract for `host/application/framework`

Three Python layers, one explicit dependency chain:

```
axelera_runtime2  <--  axelera_zoo  <--  framework (axelera/app, ax_models/, ax_datasets/, ...)
```

An upstream change must revalidate its downstream consumers. A downstream
change stays local. This doc pins the contract so the CI stays boring.

## Workflows and what they run

| Workflow | File | Triggers |
|---|---|---|
| Framework pre-commit + tox | `.github/workflows/python.yml` | Any change under `axelera/`, `ax_*/`, `tests/`, `operators/`, `tox.ini`, `pyproject.toml`, `Makefile`, `containerless.sh`. Also when `axelera_runtime2/**` or `axelera_zoo/**` change (downstream gating in the `framework-ci` `if:`). |
| Global pre-commit (changed files) | `.github/workflows/python.yml` | Every pull request. Ungated on purpose, see below. |
| Build and test axelera_zoo | `.github/workflows/build-axelera-zoo.yml` | Any change under `axelera_zoo/**` OR `axelera_runtime2/**` (so zoo gets revalidated when rt2 changes). |
| Build and upload axelera_runtime2 wheel | `.github/workflows/build-axelera-runtime2.yml` | Any change under `axelera_runtime2/**` or `trackers/**`. Doc-only PRs no longer trigger the 3.10-3.14 matrix. |

## Path-filter outputs (`python.yml` `changes` job)

Each output becomes the `if:` gate on downstream jobs:

| Output | Paths |
|---|---|
| `runtime2` | `axelera_runtime2/**` |
| `zoo` | `axelera_zoo/**` |
| `framework` | `axelera/**`, `ax_models/**`, `ax_datasets/**`, `ax_evaluators/**`, `operators/**`, `tests/**`, `tox.ini`, `pyproject.toml`, `Makefile`, `containerless.sh`, `.github/workflows/python.yml` |
| `trackers` | `trackers/**` + specific runtime2 tracker paths |
| `onnx_opt` | `ax_models/onnx_optimizations.py`, `ax_models/base_onnx.py`, `axelera/app/operators/preprocessing.py`, `tests/onnx_opt/**` |
| `axelera_llm` | `axelera_llm/**` |

`framework-ci` gates on `framework || zoo || runtime2`. `build-axelera-zoo.yml` gates on its own paths (rt2 + zoo). This means every upstream change mechanically revalidates every downstream.

## Pre-commit runs in one place, on the changed files

`precommit-global` is the only job that runs `pre-commit`, over the files the pull
request added or modified. It carries no `if:` and no `needs:`, so no path filter
can keep a pull request away from it, and `python.yml` triggers on `pull_request`
alone because that is the only event carrying the base commit the step diffs
against. Both properties matter more than they look: pre-commit reads Markdown
too (`end-of-file-fixer` and `mixed-line-ending` do; `trailing-whitespace` is the
one that excludes it), so a gate that skipped docs, or an event with no base
commit, would leave those files with no hook at all and no job going red to say
so. `tests/test_precommit_gate.py` asserts both.

Per-hook file selection belongs in `.pre-commit-config.yaml`, not in a workflow
gate, so a hook that should see a file it currently skips is a change to that
hook's `files`/`exclude`. `framework-ci` no longer runs `pre-commit --all-files`
and keeps the job name "Framework pre-commit + tox" only because that string is
reported outside this repository.

## The application checkin gate (`checkin.yml` `check-files` job)

`full-tests` runs `deploy.py` and `inference.py` over the yaml model cards, and
`perf-tests` measures the same path, so both only run when a change reaches
`axelera.app` AS THE SDKENV IMAGE BUILDS IT. The `check-files` job runs them
unless EVERY changed file that is not Markdown lives in one of these trees:

    model_axscript/  axelera_zoo/  model_recipes/  axelera_llm/
    axelera_runtime2/  worklogs/

What the image contains decides that list, and it is not the same as what this
repository contains. The image installs `axelera-rt` and `axelera-devkit` as
RELEASED wheels at `AXELERA_VERSION`, then copies this tree and does two things
with it: `pip install -e axelera_zoo`, and `make operators`. So `operators/`
changes what the checkin runs and stays on the run side, while
`axelera_runtime2/` is never installed from source and cannot change the result,
even though `axelera/app/display_cv.py` and its siblings import
`axelera.runtime2` at module scope. `framework-ci` in `python.yml` is what
validates `axelera.app` against runtime2 from source, and it already gates on
`runtime2`.

`worklogs/` is on the list for a different reason from the rest: it holds
engineering notes, and it is neither tracked nor gitignored here, so a worklog
that does get committed would otherwise buy an 86 minute checkin.

The list is a deny list. A tree that nobody has classified still runs the full
checkin, so the failure mode of forgetting to update it is a slow pull request
and never an untested product.

`tests/test_checkin_gate.py` reads the pattern and the grep flags out of the
workflow and pins the deny list as a whole. Extend both when a new decoupled
tree appears, and re-read the image recipe first: a tree becomes reachable the
day the image starts installing it.

## Zoo's two-speed test matrix

`axelera_zoo/tox.ini` splits tests by rt2 dependency:

| tox env | Installs runtime2? | Runs | Typical duration |
|---|---|---|---|
| `py310-zoo` | No | Every test module, but 2 (`test_loader.py`, `test_pipelines_ultralytics.py`) auto-skip via `pytest.importorskip('axelera.runtime2.op')` at top. Net: 6 test files run. | ~5 s after pip install |
| `py310-rt2` | Yes (editable + opencv-headless swap) | Every test module (all 8 files collect and run). | ~60-90 s |

New tests that need `from axelera.runtime2 import op` should use `op = pytest.importorskip('axelera.runtime2.op')` at the top of the test file instead of a direct `from axelera.runtime2 import op`. This makes the file gracefully skip when runtime2 isn't installed (py310-zoo env) and run normally when it is (py310-rt2 env). No pytest marker registration needed.

## Local reproduction

```bash
# Each layer standalone
cd axelera_runtime2 && tox -e py310-test
cd axelera_zoo && tox -e py310-zoo       # fast
cd axelera_zoo && tox -e py310-rt2       # full integration
cd .. && tox -e py310                    # framework

# Containerless dev (installs all layers editable in one venv)
source containerless.sh
```

## Adding a new sibling package

Say you add `axelera_foo/` with the same layering as zoo:

1. Add a `foo: 'axelera_foo/**'` entry to `python.yml`'s `filters:` block and a `foo: ${{ steps.filter.outputs.foo }}` output.
2. If foo is *downstream* of an existing layer, add that layer's output to foo's workflow `if:` gate (e.g., `if: needs.changes.outputs.foo == 'true' || needs.changes.outputs.zoo == 'true'`).
3. If foo is *upstream* of framework, extend `framework-ci`'s gate: `framework || zoo || runtime2 || foo`.
4. If foo has its own workflow (`build-axelera-foo.yml`), make sure its `paths:` trigger includes all upstream layers too.

## Known gaps (not in scope here)

- **Wheel-artifact handoff.** Zoo currently installs rt2 via `pip install -e ../axelera_runtime2` (sibling path). Framework installs zoo the same way. Real hand-off validation requires building an artifact in the upstream workflow and downloading it in the downstream. Tracked as a future improvement; today's monorepo shortcut works.
- **`full-tests / app-framework-test`** Docker image (from software-platform's `app_framework` image recipe) needs `pip install axelera_zoo` added. Outside this repo's scope — fix via a PR to software-platform.
