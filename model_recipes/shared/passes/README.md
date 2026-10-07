# Pre-compile passes

Every rewrite in this directory is a compiler gap, written down.

Ideally the compiler handles whatever you export: you compile, and nothing sits
between the two. In practice there are shapes it has not learned yet, for
reasons that are technical and, in every case here, temporary. A pre-compile
pass is one of those gaps made explicit: a small Python function that rewrites
the model into a shape the compiler handles well, without changing what the
model computes. It runs inside the recipe, before quantization.

Three things follow, and they are the whole design.

**Each one is temporary.** Every pass names what would retire it, in its
`RETIREMENT_CONDITION`. When the compiler learns the shape, the file is deleted
and the recipes that named it stop naming it. This directory should shrink over
releases, not grow.

**Each one is reproducible.** Working out *which* rewrite an awkward model needs
is the hard part, and it is easy to do once by hand and lose. A pass is that
work written down, with a check that says the rewritten model still computes
what the original did, so the next person runs it and gets the same model
rather than the same puzzle.

**You can write your own.** Not every reason to rewrite a model is ours. Yours
might be a shape our compiler has not met, or something particular to your
deployment. Section 4 is the anatomy; nothing here is reserved for passes we
wrote.

## 1. Does my model need one?

**If your build stops**, look at what the message names. A pass is the answer
when it names an operator, or an attribute of one, rather than anything about
your model: a node type the compiler will not lower, or a flag on a layer it
cannot express. The exact wording comes from the compiler and changes between
releases, so match on the operator name rather than on the sentence.

Take that operator name to the table below. If it is there, add that pass and
build again. If it is not, section 4 is how you write one.

**If your build succeeds**, you are not done reading. A model can compile
perfectly and still spend most of its time in one layer that the compiler had
to lower conservatively, because the shape it was handed does not map onto the
chip's matrix unit. Nothing warns you about this. `gemm_to_conv` is the case we
ship: it rewrites a large trailing fully-connected layer so the compiler can
split it across the matrix array, and it deliberately does nothing when the
layer is small enough not to matter. Benchmark first, and reach for it when a
classifier's last layer is where the time goes.

Most models need no pass at all.

## 2. What ships today

| Pass | Without it | What it does | Weights | Backends |
| --- | --- | --- | --- | --- |
| `replace_focus_layer` | the build stops: the compiler cannot lower the Focus stem | Some YOLO models start by cutting the image into four half-size copies and stacking them before the first convolution. The pass folds that into the one convolution it is equivalent to. | `.onnx` | `axrelay` |
| `gemm_to_conv` | the model builds and runs, with the last layer using a fraction of the matrix array | Some classifiers end in one large matrix multiply. Rewritten as convolutions the compiler can split across the array, the same arithmetic uses more of the chip. Skips itself on layers small enough not to matter. | `.onnx` | `axrelay` |
| `disable_ceil_mode_in_avgpool` | the build stops: the codegen has no ceil-mode pooling window | Some models average-pool with a window that hangs off the edge of the feature map (`ceil_mode=True`). The pass clears the flag. | `.pt2` | `axrelay`, `axgraph` |

This table is the whole list. A test fails if a file here has no row, or a row
here has no file, so it cannot go stale as passes are added and retired.

The "without it" column is the one to read first, because the passes are not
all the same kind of thing. Two of these announce themselves by stopping your
build. `gemm_to_conv` does not: that model compiles, runs, and is simply slower
than its arithmetic requires, which is a thing no error message will ever tell
you. Two caveats on it, therefore: no model in the Axelera Model Zoo declares
it today, so it carries no published before/after number and should be treated
as a lead to benchmark rather than a setting to switch on; and it has a shape
it declines to touch, so naming it on a model whose last layer is already small
is a no-op rather than a mistake.

## 3. How to use one

Add its name to `PRE_COMPILE_PASSES` at the top of your recipe:

```python
PRE_COMPILE_PASSES = [
    'replace_focus_layer',
]
```

That is all. The recipe already imports `apply` from this directory and calls
it before quantizing, and it prints each rewrite as it runs:

```
[stage] pre_compile_passes
[pass] replace_focus_layer: YOLO Focus (space-to-depth) + Conv stem folded into one Conv  (shared/passes/replace_focus_layer.py)
```

A pass that looks at your model and finds nothing to rewrite says so on a
second line, and the build carries on:

```
[pass] replace_focus_layer: found nothing to rewrite in this model
```

That is the line to read when you want to know whether a workaround actually
fired, rather than assuming it did because the build passed. A pass declared for
a weight format your build is not using is skipped and says so; that is not an
error either. Remove a name to drop the rewrite, and set the list to `[]` to run
none.

This directory is part of `shared/`, which has to sit beside your recipe or one
level up. `axzoo build` and `axzoo fork --script` both put it there, and a
recipe imports it before it does anything else, so a copy made without it stops
at once and names the directory rather than failing later on something obscure.
If you copy a recipe out by hand, copy `shared/` with it and the rewrites come
along: they used to be a separate directory you could forget on its own.

## 4. How to write your own

A pass is a module with a `run` function and six declarations. Nothing else,
and nothing imported from the model zoo: this directory depends only on what
the compiler already installs.

Most things worth rewriting fall into a few shapes: an operator the compiler
cannot place, a layer written so it uses a fraction of the hardware, a set of
operators it supports individually but groups badly, or a graph that will not
fit the memory it is given. Yours does not have to be any of those. A rewrite
you want for your own reasons, on your own model, is a perfectly good pass.

Two of those are worth thinking about rather than copying. `WITHOUT_IT` is what
a reader checks their own symptom against, so write the symptom, not the
diagnosis; and if the answer is "nothing visible", say so, because then the
table is the only way anyone finds your pass. `RETIREMENT_CONDITION` is what
lets this file be deleted later by someone who was not you.

Finding the rewrite is usually harder than writing it, and it is a good use of
a coding assistant: describe the graph, the error, and what the hardware wants,
and iterate on candidate rewrites against the equivalence check below. What
makes the result worth keeping is that it lands here as a file with that check
attached, so the next person reproduces your model instead of your search.

All six are read by parsing this file, never by importing it, so each one has
to be a **literal**. A value computed from anything else is invisible to that
reader and the pass is refused.

```python
"""One paragraph: what the build looks like without this pass."""

SUPPORTS = ('axrelay',)                  # compiler backends it is valid for
SOURCE_KINDS = ('.onnx',)                # weight formats it operates on
WITHOUT_IT = 'what goes wrong if nobody runs this'
DESCRIPTION = 'one line, printed when it runs'
RETIREMENT_CONDITION = 'what would let us delete this'
DEPRECATED_IN = None                     # an SDK version, once absorbed


def run(model):
    """Rewrite `model` and return it, or return None if there is nothing to do."""
    return None
```

`None` is how a pass says it found nothing to rewrite, and it is what makes the
build print that rather than claim a rewrite. Return it whenever your detector
does not match, or your threshold says the rewrite would not pay. Do not return
the model to mean the same thing: a pass that edits its model in place returns
that same object when it *did* act, so the two cannot be told apart.

Save it as `<name>.py` here, add `'<name>'` to your recipe's
`PRE_COMPILE_PASSES`, add a row to the table in section 2, and build.

`replace_focus_layer.py` is the worked example, and it is worth reading in
full, because the three parts it splits into are the three parts any pass has:

1. **Detect.** `detect_focus_conv_pattern` walks the graph and returns the
   nodes of the pattern, or `None`. It matches on structure, never on node
   names: a name comes from whichever exporter wrote the file and changes when
   someone re-exports.
2. **Rewrite.** `transform_focus_conv_weights` and `replace_focus_layer` build
   the replacement and splice it in. A pass that finds no match returns `None`
   rather than raising, so naming it on a model that does not need it costs
   nothing and says so.
3. **Prove it.** `run` compares the original and the rewritten model on a
   random input, through `outputs_equivalent` in `_onnx.py`, and raises when
   they disagree.

   Read that helper before relying on it. It **fails open**: if onnxruntime is
   not installed, or errors on your graph, it returns `True` and the build
   continues unchecked. That is deliberate, so a rewrite is never blocked by an
   infrastructure problem, but it means a green build is not by itself proof
   the check ran. If your pass rewrites weights, run its equivalence check once
   by hand, with onnxruntime present, and confirm it can also fail.

Step 3 is the one to copy even if you copy nothing else. A rewrite that is
subtly wrong does not fail the build; it produces a model that compiles, runs
fast, and gets the answer wrong, and you find out at accuracy-evaluation time
with nothing pointing back here. `disable_ceil_mode_in_avgpool` is the
exception that shows the rule: it deliberately changes the arithmetic, so there
is nothing to assert equal, and its docstring says so and hands the check back
to you.

## 5. When does it go away

Every pass here exists because the compiler cannot yet do something, and each
one names the thing that would retire it in its `RETIREMENT_CONDITION`.

When a release absorbs the fix, the pass is not deleted straight away. Its
`DEPRECATED_IN` is set to that SDK version, and for one release it is skipped
with a message naming the version that absorbed it, so a recipe that still
lists it keeps building:

```
[pass] replace_focus_layer: absorbed into the compiler in SDK 2.1, skipped. Remove it from PRE_COMPILE_PASSES.
```

Remove the name when you see that. The release after deletes the file from the
SDK's own copy of this directory, and a recipe in that tree which still names it
then fails on its next build, with the list of passes that do exist. The failure
comes when the pass stage runs, which is after the weights are fetched, not at
startup.

A copy of this directory that you took with you (`axzoo fork --script` writes
one, and `axzoo build` writes one beside its output) is yours: no release
touches it, so a pass retired upstream keeps running there until you delete the
file. That is the same property that lets you edit these rewrites at all. Two
things follow. `axzoo build` replaces the directory beside its own default
output wholesale, so a retired pass never lingers there. In a directory you
chose with `--output`, or forked into, it stops rather than overwrite a rewrite
you edited, and tells you to run the forked recipe itself
(`python <recipe>.py --build`) if the edited rewrite is the one you want.
