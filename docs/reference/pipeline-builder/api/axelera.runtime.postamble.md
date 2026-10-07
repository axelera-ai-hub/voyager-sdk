---
title: "axelera.runtime.postamble"
---
# `axelera.runtime.postamble`


The `Postamble` object handed to `op.load` (see the `Postamble` class).

## Summary

| Name | Description |
|------|-------------|
| [Postamble](#postamble) | Names a fast kernel for `op.load` to run a model's *postamble*. |

---

### Postamble

Names a fast kernel for `op.load` to run a model's *postamble*.

When a model is compiled, most of the network runs on the AIPU, but the
compiler leaves a trailing piece as a small graph in the `.axm` (its
`postamble_graph`): the model-specific decode or activation that turns the
device's output into the model's declared results: for example YOLO box
decoding, or a softmax. That graph is the *postamble* (the mirror of the
*preamble* the compiler leaves in front of the network).

`op.load` runs the postamble on onnxruntime unless you hand it a
`Postamble` naming a kernel to run in the graph's place. Two ways to feed
that kernel, trading effort for speed:

- `kernel=MyKernel`: simplest. The runtime unpads the AIPU's raw output and
  dequantizes it to float tensors, and your kernel runs on those.
- `fused_kernel=MyKernel`: fastest. Your kernel is handed the AIPU's raw
  int8 output and does the unpad and dequantize itself, fused with the decode
  in one pass (how the built-in YOLO kernels avoid building large float
  tensors).

An empty `Postamble()` asks for the onnxruntime postamble *and* suppresses the
automatic choice. Omitting the argument leaves that choice open, so a model whose
`.axm` carries framework metadata gets the matching kernel from `.optimized()`.
Those are different requests, which is why this class defines no `__bool__`: an
empty `Postamble` is a statement of intent, not the absence of one, and code that
tested it for truth would silently treat it as "no opinion".

Name the kernel as a class, as the bare name of a built-in kernel
(`'ClassificationPostamble'`), or as a `"module:ClassName"` string for a kernel
of your own, which `op.load` imports. A class is stored as its import path, so a
`Postamble` always holds a name a generated recipe can import. Any `**kwargs`
are the kernel's own parameters (e.g. `num_classes`), passed to it when it is
built.

**Raises:**

- **ValueError**: if both `kernel` and `fused_kernel` are given; if kernel parameters are given with neither slot to receive them; or if a kernel class cannot be named importably (defined in `__main__`, or inside a function).

**Constructor:**

```python
__init__(kernel: Any = None, fused_kernel: Any = None, **kwargs: Any)
```

**Methods:**

#### as_spec

```python
as_spec() -> dict[str, Any]
```

Return this Postamble as plain data, for a manifest or a recipe.

An empty Postamble is an empty mapping, not an absent one: the two are
different requests, and only the caller's own choice belongs in a package.

#### classmethod from_spec

```python
from_spec(spec: dict[str, Any]) -> Postamble
```

Rebuild a Postamble from `as_spec` output.
