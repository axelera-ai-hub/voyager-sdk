#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""Export a @publicdemo function from rt_demo.py into a standalone example.

The generated file lives under examples/pipeline_builder/ and depends only on
axelera.runtime + argparse, so end users can run it as a starting point for
their own pipelines without pulling in the smoke-test scaffolding.

Usage:
    python tools/export_pipeline_builder_demo.py                 # list public demos
    python tools/export_pipeline_builder_demo.py classification  # export a single demo
    python tools/export_pipeline_builder_demo.py all             # export every @publicdemo

The first invocation that targets a concrete demo writes
    examples/pipeline_builder/<name>.py

The body of the demo is taken verbatim from the @publicdemo function in
internal-apps/axelera/internal/apps/rt_demo.py, with smoke-test-specific
plumbing (Throttle, args.dump/LOG, inputs(), emit, etc.) stripped out and
replaced by axelera.runtime equivalents.
"""

from __future__ import annotations

import argparse
import ast
import builtins
from dataclasses import dataclass
import importlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERNAL_APPS_DIR = REPO_ROOT / 'internal-apps'
RT_DEMO_PY = INTERNAL_APPS_DIR / 'axelera' / 'internal' / 'apps' / 'rt_demo.py'
# Default output directory for generated examples.  Centralised here because
# the example tree may be relocated as the runtime2 layout evolves; the rest
# of this tool references EXAMPLE_DIR rather than the literal path.
EXAMPLE_DIR = REPO_ROOT / 'examples' / 'pipeline_builder'

# axelera.runtime top-level names we resolve attribute chains against.  An
# import is emitted for each one whose attributes the demo actually touches.
KNOWN_RUNTIME_BASES = ('op', 'cv', 'display', 'scheduler')

# Textual substitutions applied to each demo body in order before the
# close-check is injected.  Anything that is purely smoke-test scaffolding
# (Throttle, args.dump/LOG block) is rewritten to the empty string; runtime
# helpers that only exist inside rt_demo (inputs/emit/_get_frame_name) and
# the smoke-side emit hook on vis() are rewritten to their standalone
# equivalents.  Order matters: `inputs(args)` must be rewritten before the
# close-check regex looks for `cv.create_source(args.input)`.
BODY_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Smoke-side fps-limiter; the user-facing template uses the default fps.
    (re.compile(r'^\s*Throttle\(args\.fps_limit\),\s*\n', re.MULTILINE), ''),
    # `if args.dump: LOG.info(...)` debug blocks (incl. multi-line LOG follow-ups).
    (re.compile(r'^([ \t]*)if args\.dump:[ \t]*\n(?:\1[ \t]+LOG\..*\n)+', re.MULTILINE), ''),
    # Smoke-runner frame source -> standalone cv.create_source.  Replacing the
    # call (not the whole `for` line) preserves both bare iteration and
    # `enumerate(inputs(args))`.
    (re.compile(r'inputs\(args\)'), 'cv.create_source(args.input)'),
    # emit() is the smoke stdout channel; _get_frame_name reads a smoke-side map.
    # `(?<![\w.])` excludes attribute calls (e.g. `obj.emit(...)`) -- `\b`
    # alone would treat the boundary between `.` and `emit` as a word break.
    (re.compile(r'(?<![\w.])emit\('), 'print('),
    (re.compile(r'_get_frame_name\(img\)'), "'frame'"),
    # Smoke-side progress bar: rt_demo wraps the stream iterable in
    # `progress = _create_progress(...)` / `progress(<it>)` (alive_progress).
    # The standalone template depends only on axelera.runtime + argparse, so
    # drop the assignment and unwrap the call to iterate <it> directly.  The
    # `(?<![\w.])` lookbehind keeps the unwrap from matching `_create_progress(`.
    # Order: the assignment line is removed first so only the wrapper call
    # remains for the unwrap.  Locals that only fed the progress title
    # (title/fps/total_frames) are dropped afterwards by _strip_dead_locals.
    (re.compile(r'^[ \t]*progress = _create_progress\(.*\)\n', re.MULTILINE), ''),
    (re.compile(r'(?<![\w.])progress\((.+)\)'), r'\1'),
    # rt_demo's `load_model(args, <axm>, <onnx>)` picks .axm or .onnx based on
    # --onnx; the standalone template has no --onnx flag, so collapse it to the
    # .axm path: op.load(<axm>).  The first arg is the .axm path (no comma inside
    # the f-string), the rest (the .onnx path) is dropped.
    (re.compile(r'load_model\(\s*args\s*,\s*([^,]+),\s*[^)]+\)'), r'op.load(\1)'),
)


# Match `vis(img, <name>)` where <name> is any identifier.  Used in
# classification-style demos to neutralise the smoke wrapper's emit hook
# (see _neutralise_vis_result).  Matching by structure (any identifier)
# instead of by the demo-author's specific local name (`top5`, `topk`)
# means a future @publicdemo that names its result `result` or `top5k`
# is still rewritten correctly.
_VIS_RESULT_RE = re.compile(r'vis\(img,\s*([A-Za-z_][A-Za-z0-9_]*)\)')


_RT_DEMO_MODULE_CACHE = None


def _load_rt_demo_module():
    """Import ``axelera.internal.apps.rt_demo`` from the internal-apps tree.

    Loaded as a proper package (rather than via spec_from_file_location) so
    that ``rt_demo.py``'s relative imports (``from .tools_utils import ...``)
    resolve.  Cached because the module registers operator subclasses on
    import via ``__init_subclass__``; re-running import would raise on the
    second call (e.g. for the 'all' export target that processes each
    @publicdemo in turn).
    """
    global _RT_DEMO_MODULE_CACHE
    if _RT_DEMO_MODULE_CACHE is not None:
        return _RT_DEMO_MODULE_CACHE
    internal_apps = str(INTERNAL_APPS_DIR)
    if internal_apps not in sys.path:
        sys.path.insert(0, internal_apps)
    _RT_DEMO_MODULE_CACHE = importlib.import_module('axelera.internal.apps.rt_demo')
    return _RT_DEMO_MODULE_CACHE


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    """Return the AST FunctionDef for the @publicdemo named ``name``."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(f'function {name!r} not found')


def _attribute_chain(node: ast.AST) -> str | None:
    """Return ``a.b.c`` for an Attribute(Name) chain, else None."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return '.'.join(reversed(parts))
    return None


def _import_bindings(node: ast.Import | ast.ImportFrom) -> list[tuple[str, str]]:
    """Return ``[(bound_name, rendered_source), ...]`` for each alias on
    ``node``.  Shared between the bound-name collector (which only cares
    about the names) and the import propagator (which needs the source
    text), so both agree on the alias-resolution rule.

    ``Import``: ``import a.b.c`` binds ``a`` (the top-level package);
    ``import a.b.c as foo`` binds ``foo``.
    ``ImportFrom``: ``from x import y`` binds ``y``; ``... as z`` binds ``z``.
    Caller is responsible for filtering relative ImportFrom (``node.level > 0``).
    """
    if isinstance(node, ast.Import):
        return [
            (
                a.asname or a.name.split('.', 1)[0],
                f'import {a.name} as {a.asname}' if a.asname else f'import {a.name}',
            )
            for a in node.names
        ]
    return [
        (
            a.asname or a.name,
            (
                f'from {node.module} import {a.name} as {a.asname}'
                if a.asname
                else f'from {node.module} import {a.name}'
            ),
        )
        for a in node.names
    ]


def _propagated_imports(tree: ast.Module, body_loads: set[str]) -> list[str]:
    """Render every module-level ``import`` / ``from ... import ...`` in
    ``tree`` whose bound name appears in ``body_loads``, EXCEPT relative
    imports and ``axelera.runtime`` (already emitted by the template's
    own runtime-imports line via :class:`BodyAnalysis.runtime_bases`).

    Lets demo bodies use stdlib / third-party names hoisted to the top
    of rt_demo.py (e.g. ``Counter`` for tracking) without the export
    silently producing a ``NameError``.
    """
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (
            node.level > 0 or node.module == 'axelera.runtime'
        ):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for bound, src in _import_bindings(node):
                if bound in body_loads:
                    out.append(src)
    return out


@dataclass(frozen=True)
class _BodyAnalysis:
    """Facts collected in one ``ast.walk(fn_node)`` pass."""

    loads: set[str]
    runtime_bases: set[str]
    has_vis_text: bool


def _analyze_body(fn_node: ast.FunctionDef) -> _BodyAnalysis:
    """Single-walk collector for everything ``export_demo`` needs from the
    demo body: which names it loads (for import propagation + undefined
    diagnostics), which axelera.runtime bases it touches (for the runtime
    imports line), and whether it renders labels via ``vis.text(...)``
    (so the wrapper-emit `vis(img, X)` call is neutralised on export).
    """
    loads: set[str] = set()
    runtime_bases: set[str] = set()
    has_vis_text = False
    for sub in ast.walk(fn_node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            loads.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            chain = _attribute_chain(sub)
            if chain is not None:
                head = chain.split('.', 1)[0]
                if head in KNOWN_RUNTIME_BASES:
                    runtime_bases.add(head)
        elif isinstance(sub, ast.Call) and _attribute_chain(sub.func) == 'vis.text':
            has_vis_text = True
    return _BodyAnalysis(loads=loads, runtime_bases=runtime_bases, has_vis_text=has_vis_text)


def _docstring(fn_node: ast.FunctionDef) -> str:
    """Return the docstring of fn_node (without surrounding quotes), or ''."""
    return ast.get_docstring(fn_node) or ''


def _extract_body_source(fn_node: ast.FunctionDef, full_source: str) -> str:
    """Return the body source of fn_node, with docstring stripped.

    Lines are kept verbatim (preserving comments and formatting) by slicing
    the original source; only the docstring expression is removed.
    """
    body = list(fn_node.body)
    # Drop a leading docstring expression.
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        return ''
    first_lineno = body[0].lineno
    last_lineno = body[-1].end_lineno
    lines = full_source.splitlines(keepends=True)
    chunk = ''.join(lines[first_lineno - 1 : last_lineno])
    return chunk


# The demo's main input loop, matched to inject the window-close check.
# Two shapes occur: bare iteration over `cv.create_source(args.input)`
# (image/per-frame demos) and `<pipeline>.stream(<source>)` (streamed demos,
# where the source comes from a `with cv.create_source(...) as source:`).
_STREAM_FOR_RE = re.compile(
    r'^(?P<indent>[ \t]*)for [^\n]+'
    r'(?:cv\.create_source\(args\.input\)|\.stream\([^\n]*\))'
    r'[^\n]*:[ \t]*\n',
    re.MULTILINE,
)


def _neutralise_vis_result(body_src: str) -> str:
    """Rewrite every ``vis(img, <name>)`` to ``vis(img, None)``.

    Only safe to call when the demo renders labels manually (signalled
    by :attr:`_BodyAnalysis.has_vis_text`).  Matching by identifier
    structure rather than the literal ``top5``/``topk`` chosen by the
    current classification demos means a future @publicdemo that names
    its result differently still exports correctly.  Existing
    ``vis(img, None)`` calls are idempotently rewritten back to
    themselves, so no special case is needed.
    """
    return _VIS_RESULT_RE.sub('vis(img, None)', body_src)


def _inject_close_check(body_src: str) -> str:
    """Insert ``if vis.is_closed: break`` as the first statement of each
    main input loop (see :data:`_STREAM_FOR_RE`), so the standalone demo
    exits when the user closes the display window (Q/ESC/Space in the
    opencv backend).

    Mirrors the same check rt_demo.py applies inside ``inputs()``.
    """

    def _add_check(match: re.Match) -> str:
        body_indent = match.group('indent') + '    '
        return f'{match.group(0)}{body_indent}if vis.is_closed:\n{body_indent}    break\n'

    return _STREAM_FOR_RE.sub(_add_check, body_src)


def _transform_body(body_src: str, *, has_vis_text: bool) -> str:
    """Rewrite a @publicdemo body for standalone use.

    Applies BODY_SUBS in order (strip smoke scaffolding, swap rt_demo
    helpers for runtime equivalents); when ``has_vis_text`` is true the
    demo renders labels manually, so also neutralises the smoke wrapper's
    emit hook (``vis(img, X)`` → ``vis(img, None)``); finally injects
    ``if vis.is_closed: break`` at the top of each ``cv.create_source``
    loop so pressing Q/ESC/Space stops inference -- mirroring the same
    check rt_demo.py applies inside ``inputs()`` for its hosted demos.
    """
    for pat, repl in BODY_SUBS:
        body_src = pat.sub(repl, body_src)
    if has_vis_text:
        body_src = _neutralise_vis_result(body_src)
    return _inject_close_check(body_src)


def _strip_dead_locals(body_src: str) -> str:
    """Drop ``<name> = <expr>`` whose target is never loaded in ``body_src``.

    Stripping smoke scaffolding can orphan the locals that only fed it --
    removing the progress bar in ``detection_stream`` leaves the
    ``title``/``fps``/``total_frames`` lines that only built its title.
    Only assignments with a single ``Name`` target and a call-free (hence
    side-effect-free) RHS are removed, so deletion cannot change behaviour;
    ``op.seq(...)`` and friends are kept because their RHS contains a call.
    Iterates to a fixpoint so a chain (``title`` -> ``fps``) collapses fully.

    ``body_src`` must be dedented to column 0 so it parses as a module.
    """
    while True:
        tree = ast.parse(body_src)
        used = {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        dead_lines: set[int] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id not in used
                and not any(isinstance(s, ast.Call) for s in ast.walk(node.value))
            ):
                dead_lines.update(range(node.lineno, node.end_lineno + 1))
        if not dead_lines:
            return body_src
        lines = body_src.splitlines(keepends=True)
        body_src = ''.join(ln for i, ln in enumerate(lines, 1) if i not in dead_lines)


def _dedent_into_main(body_src: str, indent: str = '        ') -> str:
    """Re-indent demo body (originally 4-space) under the with/for in main()."""
    dedented = textwrap.dedent(body_src)
    # Strip leading/trailing blank lines so the rendered file is tidy.
    dedented = dedented.strip('\n')
    # Dead-code sweep runs here (on the dedented, module-parseable body)
    # rather than in _transform_body, which works on still-indented text.
    dedented = _strip_dead_locals(dedented)
    # Removing a dead assignment can strand a blank line at the top of its
    # block (e.g. between `with ... as source:` and the loop).  black keeps
    # blank lines after non-def block openers, so collapse them here.
    dedented = re.sub(r'(:[ \t]*\n)(?:[ \t]*\n)+', r'\1', dedented)
    return textwrap.indent(dedented, indent)


TEMPLATE = '''\
#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""{docstring}

Auto-generated from rt_demo.py's @publicdemo `{name}` by
tools/export_pipeline_builder_demo.py.  Edit the original demo in rt_demo.py
rather than this file -- regenerate with
`python tools/export_pipeline_builder_demo.py {name}`.

Usage:
    python {name}.py PATH [--display opencv|console|none|auto] [--no-wait]
"""

from __future__ import annotations

import argparse
from pathlib import Path

{imports_block}

# Populate this directory once with the .axm files referenced below
# (`axdownloadmodel --axm <name>` inside this directory).
MODEL_DIR = str(Path.home() / ".cache" / "axelera" / "runtime2")


def main(args):
    with display.App(renderer=args.display) as visualizer:
        vis = visualizer.create_window({title!r}, (args.window_width, args.window_height))
{body}
        # Headless backend ('none') never sets is_closed -- wait_for_close
        # would block forever -- so skip the wait there even when --wait is set.
        if args.wait and args.display != "none":
            vis.wait_for_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description={short_desc!r})
    parser.add_argument("input", help="Image or video file to process")
    parser.add_argument(
        "--display",
        choices=["none", "opencv", "console", "iterm2", "auto"],
        default="auto",
        help="Display backend (default: auto)",
    )
    parser.add_argument("--window-width", type=int, default=800)
    parser.add_argument("--window-height", type=int, default=500)
    parser.add_argument(
        "-w", "--wait", action=argparse.BooleanOptionalAction, default=True,
        help="Keep the window open after processing until the user closes "
             "it (default). Use --no-wait to exit as soon as the input ends; "
             "useful for batch/video runs where you do not need to inspect "
             "the final frame.",
    )
    main(parser.parse_args())
'''


_BUILTIN_NAMES = frozenset(dir(builtins))


def _undefined_loads(tree: ast.Module) -> list[tuple[int, str]]:
    """Return ``(lineno, name)`` for every ``Name(Load)`` in ``tree`` whose
    id is neither bound somewhere in the file nor a builtin.

    One ``ast.walk(tree)`` pass collects both sides at once: ``bound``
    grows from every assignment target / import / def name / function or
    lambda parameter / exception handler; ``loads`` accumulates every
    ``Name(Load)`` node so we can preserve linenos for the error message.

    The historical failure mode this guards against: a @publicdemo body
    references a module-level helper in rt_demo (e.g. ``render_branch``
    for ``detection_branches``).  The exporter only carries the function
    body, so the helper does not follow, and the generated file parses
    fine but raises ``NameError`` at first run.  Catching it here turns
    a runtime regression into an export-time failure.

    Treats the file as a single scope; that's an over-approximation but
    the generated file is flat (module-level imports + constants +
    ``main(args)`` + ``__main__`` block), so the only undefined-load
    cases that escape detection are ones that genuinely shadow nothing.
    """
    bound: set[str] = set()
    loads: list[ast.Name] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                loads.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for name, _src in _import_bindings(node):
                bound.add(name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        # Function / lambda parameters (ast.arg, not Name).  Independent
        # branch because FunctionDef/AsyncFunctionDef may be matched by
        # the def-name branch above; Lambda only lands here.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            for a in args.posonlyargs + args.args + args.kwonlyargs:
                bound.add(a.arg)
            if args.vararg:
                bound.add(args.vararg.arg)
            if args.kwarg:
                bound.add(args.kwarg.arg)
    return [(n.lineno, n.id) for n in loads if n.id not in bound and n.id not in _BUILTIN_NAMES]


def _validate_generated(path: Path) -> None:
    """Parse the generated file and reject it if any names are unresolved.

    Syntax errors come first; without a parse we cannot reason about
    names.  Unresolved name loads then fail the export because they
    almost always indicate the demo body depends on an rt_demo
    module-level helper the exporter does not carry over (see
    ``_undefined_loads``).
    """
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise SystemExit(f'{path}: generated file does not parse: {e}')
    undefined = _undefined_loads(tree)
    if undefined:
        listing = '\n  '.join(f'L{ln}: {nm!r}' for ln, nm in undefined)
        raise SystemExit(
            f'{path}: undefined name(s) in generated file -- the exporter '
            f'likely missed a helper that the @publicdemo body depends on. '
            f'Either inline the helper into the demo body in rt_demo.py, '
            f'or demote the demo to @demo. Found:\n  {listing}'
        )


def _format_template(*, name: str, docstring: str, imports_block: str, body: str) -> str:
    """Render the standalone-example template for one demo."""
    return TEMPLATE.format(
        docstring=docstring.strip(),
        name=name,
        title=name,
        short_desc=docstring.splitlines()[0] if docstring else f'{name} demo',
        imports_block=imports_block,
        body=body,
    )


def _format_with_black(source: str, path: Path) -> str:
    """Pass ``source`` through ``black`` so generated examples land
    pre-formatted in CI-compliant style.  black is the formatter the
    repo's ``.pre-commit-config.yaml`` runs on everything under
    ``examples/`` and ``tools/`` (ruff-format is scoped to
    ``axelera_runtime2/``).  Without this pass, every regeneration
    would produce a file pre-commit would then immediately rewrite.

    ``--stdin-filename`` lets black discover the project's
    ``pyproject.toml`` ``[tool.black]`` block via the file's eventual
    location even though we feed source on stdin.

    Fails open: if black is missing or returns non-zero, emit a one-line
    warning and return the unformatted source.  The file is still valid
    Python; only the style differs from the configured black rules.
    """
    if shutil.which('black') is None:
        print('  warning: black not on PATH; writing unformatted source', file=sys.stderr)
        return source
    result = subprocess.run(
        ['black', '--stdin-filename', str(path), '--quiet', '-'],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or '').strip().splitlines()
        msg = detail[-1] if detail else '(no output)'
        print(
            f'  warning: black exit {result.returncode}: {msg}; writing unformatted',
            file=sys.stderr,
        )
        return source
    return result.stdout


def list_public_demos() -> list[str]:
    """Return the names of @publicdemo functions defined in rt_demo.py."""
    mod = _load_rt_demo_module()
    return sorted(n for n, fn in mod.demos.items() if getattr(fn, 'public', False))


def _read_rt_demo_source() -> tuple[str, ast.Module]:
    """Read and parse rt_demo.py.  Cached by callers across an export-all run."""
    source = RT_DEMO_PY.read_text()
    return source, ast.parse(source)


def export_demo(
    name: str,
    output_path: Path | None = None,
    *,
    source: str | None = None,
    tree: ast.Module | None = None,
) -> Path:
    """Generate a standalone example from the @publicdemo named ``name``.

    Returns the path of the generated file.  ``source`` / ``tree`` may be
    supplied to share the parse across many calls (see ``main()``); when
    omitted they are read/parsed on demand for one-shot use.
    """
    mod = _load_rt_demo_module()
    if name not in mod.demos:
        raise SystemExit(f'unknown demo {name!r}; see --list')
    if not getattr(mod.demos[name], 'public', False):
        raise SystemExit(
            f'demo {name!r} is not marked @publicdemo; only public demos are exportable'
        )

    if source is None or tree is None:
        source, tree = _read_rt_demo_source()
    fn_node = _function_node(tree, name)
    analysis = _analyze_body(fn_node)
    docstring = _docstring(fn_node)
    body_src = _extract_body_source(fn_node, source)
    body_src = _transform_body(body_src, has_vis_text=analysis.has_vis_text)
    body_under_main = _dedent_into_main(body_src)
    # cv and display are always referenced by the template scaffolding
    # (`cv.create_source(args.input)`, `display.App(...)`); op is conditional.
    runtime_bases = analysis.runtime_bases | {'cv', 'display'}
    runtime_imports = [b for b in KNOWN_RUNTIME_BASES if b in runtime_bases]
    # Propagate any non-runtime module-level import in rt_demo.py whose
    # bound name the demo body actually references (e.g. `Counter`).
    extra_imports = _propagated_imports(tree, analysis.loads)

    imports_block = f'from axelera.runtime import {", ".join(runtime_imports)}'
    if extra_imports:
        imports_block += '\n' + '\n'.join(extra_imports)
    rendered = _format_template(
        name=name,
        docstring=docstring,
        imports_block=imports_block,
        body=body_under_main,
    )

    if output_path is None:
        output_path = EXAMPLE_DIR / f'{name}.py'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Always discover black's config from the canonical EXAMPLE_DIR
    # location (regardless of where the caller writes the output), so
    # the formatted result is deterministic.  Without this, callers that
    # redirect ``output_path`` outside the repo (notably the drift-guard
    # pytest test, which writes into a tmp dir) would lose the repo's
    # ``[tool.black] skip-string-normalization = true`` and pick up
    # default quote normalisation instead.
    rendered = _format_with_black(rendered, EXAMPLE_DIR / f'{name}.py')
    output_path.write_text(rendered)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        'demo',
        nargs='?',
        help="Name of the @publicdemo function to export, or 'all' to export every "
        '@publicdemo (omit to list).',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=None,
        help=f"Output path (default: {EXAMPLE_DIR}/<name>.py). Ignored when demo == 'all'.",
    )
    parser.add_argument(
        '--list',
        '-l',
        action='store_true',
        help='List @publicdemo functions and exit.',
    )
    args = parser.parse_args()

    if args.list or not args.demo:
        names = list_public_demos()
        print(f'Public demos ({len(names)}):')
        for n in names:
            print(f'  {n}')
        return

    if args.demo == 'all':
        if args.output is not None:
            raise SystemExit("--output is not supported with demo='all'")
        targets = list_public_demos()
    else:
        targets = [args.demo]

    source, tree = _read_rt_demo_source()
    for name in targets:
        out = export_demo(name, args.output, source=source, tree=tree)
        print(f'Wrote {out}')
        _validate_generated(out)


if __name__ == '__main__':
    main()
