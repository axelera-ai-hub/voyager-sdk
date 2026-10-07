# AxeleraScript (AxScript) — Language Guide

**Version:** 0.2 — see [Versioning](#versioning) for what a number signals.

**What this is.** A guide to the AxScript language for engineers writing kernels or models for
Axelera devices: its constructs, the execution model it exposes, and the rules that make
a program valid, with the non-obvious behaviors and limits called out. Every limit it states is
one the compiler enforces for the target you build for, and a limit can change between releases —
[§14](#14-correctness-hazards--limits) collects them.

**How to read it.** [Part I](#part-i--introduction) introduces the execution model and the vocabulary the rest of the guide relies
on; read it in order. [Part II](#part-ii--writing-programs) covers writing programs. It opens with a complete one you can run —
read that together with [§6](#6-the-datapath-model) (the datapath) and you have enough to write your own. [Part III](#part-iii--model-authoring--reference) covers
model authoring and reference material. **Holding an error message?** Go straight to [§14](#14-correctness-hazards--limits)'s
[index by symptom](#index-by-symptom--start-here-if-you-have-an-error), which maps the error string to its cause.

**Self-contained.** The guide is written to be usable on its own, without reading the compiler
source, and is kept in step with the implementation. Where it stops — a complete operator, a
tiling structure, a whole model — the reference kernels and models shipped alongside it are
worked examples of the same constructs.

---

## Part I — Introduction

This first part orients you before the reference sections: why the language exists ([§1](#1-why-axelerascript-exists)), the
model it targets ([§2](#2-the-execution-model-that-the-language-targets)), and how a program maps onto it ([§3](#3-the-programming-model-in-one-page)). It introduces the
vocabulary the rest of the guide relies on.

## 1. Why AxeleraScript exists

An Axelera device holds several **AI cores**. A core is not a general-purpose processor or a SIMT
GPU: it runs a computation by *streaming* data through a pipeline of stages — feeding data in,
computing on it, storing the result — each working on a different element at the same instant.
Performance comes from **how data flows through that pipeline** and from **keeping data close to
the compute** — most notably by keeping matrix weights
**resident** — rather than from launching many threads over shared memory.

**The language.** AxeleraScript gives you control where it matters for performance: you decide how
an operator's data flows through that pipeline, and you write it in abstractions — datapaths,
streams, and blocks such as a matrix multiply — that are more natural to program in than the
execution details beneath them. By default the compiler fills in the
mechanical parts around your structure — tiling, DMA, synchronization, and bypassing the stages
an operator does not use — but you can take control of each of them explicitly when you want to.

AxeleraScript (**AxScript**) is a **Python-embedded language**: you write what looks like a
Python function, and it is *parsed* — not executed as Python — into the intermediate
representation the compiler lowers to device instructions for an Axelera AI chip. The language is the module
`alto.script`; the `alto` package around it also holds the lowering pipeline (`alto.compiler`) and the AXOL
operator library (`alto.axol`, [§4.3](#43-operators-as-the-usual-authoring-path-scriptop--arg)). `alto.script` is imported as `A`:

```python
import axelera.compiler.alto.script as A
```

**The language is two things.** `A` supplies the programming abstractions — datapaths, unit blocks, streams,
intrinsics — and around it you write ordinary Python syntax (`for`, `if`, `with`, assignment),
which become commands just as the `A.*` constructs do. Only a subset of Python is accepted;
[§4.2](#42-the-parsed-form-and-what-it-accepts) draws the line.

**Parsed, then lowered.** The function body is read as AxeleraScript *source* and turned into
the compiler's IR — it is *not* run as ordinary Python ([§4](#4-how-your-code-is-read-and-how-it-is-packaged) covers the consequences).
**Lowering** is the compiler's step-by-step transformation of that IR toward device
instructions.

## 2. The execution model that the language targets

In the AxScript programming model, a program is a sequence of datapaths built from blocks you wire
together, and the order in which you wire them is the order data flows. What you write there
predicts how the program performs, and that is why a handful of block names appear throughout the
guide. This section defines them at the level the language works in. The limits that go with them
are collected in the [§14](#14-correctness-hazards--limits) reference.

**Datapath — a pipeline of stream units.** A *datapath* is an ordered chain of **stream units**
that run **concurrently**, wired together by **streams** (FIFOs).
Execution is a pipeline, not a sequential per-element walk — at any instant every unit is busy on
a *different* element, and how far apart they run depends on the program:

```text
one datapath at one instant — the stages are defined just below;
the main compute stage is a MatMul or a depthwise operator, set by the datapath kind

  L1 ──▶ IFD ──▶ MatMul | depthwise ──▶ DPU ──▶ ODR ──▶ L1
```

Every `──▶` *between two units* is a **stream** — a FIFO carrying elements in push order.
The first and last arrows are not streams: they are the IFD's read from memory and the ODR's
write back to it.

A single run of that chain — the pipeline starting, elements streaming through, and stopping — is
one **datapath invocation**. That is the unit of work, and it is what a
`with A.Datapath(...):` region becomes; whether a loop around one produces a single invocation or
several is [§9](#9-loops-build-time-unrolling-and-loop-lowering).

There is no point where "the datapath as a whole" processes one element and restarts. You write
each unit's work in its own block; the units then run as a pipeline. (A side-input feeder, IFD1,
can supply a second operand to the DPU in parallel — routing is
[§8.2](#82-how-the-blocks-interact--topology-and-ifd-feeder-routing).)

**The stages of a datapath and their roles** (what each does — the intrinsics each one computes are
in [§7](#7-compute-intrinsics), sizes and limits in [§14](#14-correctness-hazards--limits)). The two
middle rows are alternatives: they fill the **main compute stage**, and the datapath kind decides
which one a datapath has:

| Stage | Role |
|---|---|
| **IFD** | reads data from memory and pushes it into a stream |
| **MatMul** | on a `DP_MVMEXE` datapath: multiplies streamed vectors against a matrix held resident in the IMC (matmul / conv / linear) |
| **depthwise** | on a `DP_DWPU` datapath: depthwise sliding-window compute (depthwise conv, pooling), written through the operator library ([§6.3](#63-depthwise-operators)) |
| **DPU** | casts, activations, LUT transcendentals, elementwise math |
| **ODR** | writes results back to memory (or drops surplus positions) |

**Movement units.** Two further units move data without computing on it. They are *not* stream
units: they are not wired into a datapath's stream chain, and you write them as standalone blocks
rather than inside an `A.Datapath` ([§8.1](#81-the-block-contract-table), [§9.1](#91-the-two-for-rule)):

| Unit | Role |
|---|---|
| **CoreDMA** | copies buffers into and out of one AI core's memory; no compute |
| **SysDMA** | copies buffers between shared memories; no compute |

**Datapath kinds.** Every datapath has a `kind`, and each kind is a fixed chain of stages in
pipeline order; no kind has both a MatMul and a depthwise stage:

| Datapath (`kind`) | Stream units, in order |
|---|---|
| `A.DP_MVMEXE` (matrix multiply) | IFD0 · MatMul · DPU · ODR — plus an optional IFD1 into the DPU |
| `A.DP_DWPU` (depthwise)         | IFD0 · depthwise · DPU · ODR — plus an optional IFD1 into the DPU |

**Why the kind matters: the two kinds run in parallel.** A core runs one datapath of each kind at
a time, and each kind has its own units, so independent datapaths of different kinds can overlap:
an elementwise datapath, inferred as `DP_DWPU`, can run while a matmul (`DP_MVMEXE`) is running.
Two datapaths of the same kind always run one after the other. The overlap is opt-in: tag both
with an `id` and place their waits after both launches ([§12](#12-synchronization-and-fusion)).

You write only the stages your operator needs; the compiler fills in the rest of the chain by
**bypassing** the ones you omit ([§6.4](#64-bypass--writing-only-the-units-you-need)). And iteration lives *inside* each unit: each unit loops
over its own stream. If you do write a loop around a datapath — as [Part II](#part-ii--writing-programs)'s first program does —
the compiler moves that loop inside each unit, so the pipeline still starts once and N elements
stream through it, rather than the datapath starting and stopping N times ([§9](#9-loops-build-time-unrolling-and-loop-lowering)).

**Memory has scopes.** The memory space is not one flat space: buffers live in distinct spaces
with different **scopes**, which trade capacity against speed and degree of shared access. You
pick a buffer's scope when you allocate it ([§5.1](#51-memory-buffer-scopes--bufferscope)), and the compiler inserts
the operation that moves it between scopes as needed. The compute stages also have their own **local**
storage — the MatMul's **IMC** store and the DPU **register files** — which the datapath fills as
it runs ([§5.2](#52-allocation)).

**The matrix operand is resident.** The IMC holds one side of the multiply for as long as you
need it, and the **MatMul** in the table above is the block that streams the other side against it. In
AxScript the two appear differently: `A.IMC` is a memory scope you allocate the resident matrix
into ([§5.1](#51-memory-buffer-scopes--bufferscope)), while the multiply is written as an `A.pseudo.MatMul` block ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)).
The resident matrix is loaded **once** and stays there; loading it is a separate step — a datapath
of its own, written as a single `A.pseudo.LoadIMC` block rather than inside an `A.Datapath` — that
runs once before the multiplies. (`A.pseudo` holds the higher-level blocks that the compiler expands
during lowering: `MatMul` is a block inside a datapath, `LoadIMC` a whole datapath.) [§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)
covers both: the matmul datapath, and the IMC's tile budget. Usually that resident
matrix is a layer's weights, and the guide calls it that, but it need not be: attention makes
**K** and **V** resident and streams **Q** against them — on different cores, since one core
holds only one resident matrix at a time ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)).

**Values are vectors — the PWORD.** Every value is a fixed-width vector called a **PWORD**, of
**64 lanes**. This is why dtypes are written `"<base>x<lanes>"` (e.g. `int8x64`)
and a PWORD-packed dimension carries a trailing `/64` block dimension — the `x64` is the lane
count, not a shape you choose ([§5](#5-data-types-and-buffers)).

**The terms [§3](#3-the-programming-model-in-one-page) onward assumes:** *datapath* (a pipeline of units), *stream unit*
(IFD / MatMul / DPU / ODR), *main compute stage* (MatMul or depthwise, by datapath kind), *stream*
(a FIFO between units), *scope* (which memory a buffer lives in), *IMC* (where the resident matrix
lives), and *PWORD* (the 64-lane vector width).

## 3. The programming model in one page

With the execution model in hand, here is how an operator is actually written.

**You describe structure; the compiler schedules.** You describe the datapath *structure*
explicitly: named blocks ([§8](#8-per-unit-primitive-contracts-what-is-legal-in-which-block)) wired by streams ([§6](#6-the-datapath-model)). The mechanical parts — **data movement,
tiling, and synchronization** — are **filled in automatically by default**, each overridable
when you need control ([§8](#8-per-unit-primitive-contracts-what-is-legal-in-which-block), [§11](#11-tiling), [§12](#12-synchronization-and-fusion)). AxScript deliberately keeps the
dataflow **explicit** rather than abstracting it behind a thread/tile model, because the dataflow is
what decides performance, and that is why it reads more explicit than many kernel languages.

**Direct or composed — same language.** You either describe a datapath directly with blocks
(IFD, MatMul, DPU, ODR, streams), or wire existing operators together with ordinary SSA
assignments, which reads like a PyTorch `forward()` ([§13](#13-high-level-model-authoring)) and lowers into the datapath
implementation of each operator it calls. [Part II](#part-ii--writing-programs) works through both, and through the operator
boundary that sits between them.

**Build-time Python vs executable code.** Because the body is *parsed*, two modes coexist in it:
ordinary Python that runs **once at parse time** to compute the shapes, values, and expressions
the script is built from (metaprogramming), and the parsed AxeleraScript **statements** that
become execution instructions. The parsed part is a `@A.prim_func` body, or a `with A.script():`
block inside an ordinary Python method; the Python around it runs. Telling the two apart is what
[§4.1](#41-the-build-time--runtime-boundary) treats in full.

**A first program.** [Part II](#part-ii--writing-programs) opens with a complete program you can run: a module with one
datapath that feeds data in, computes, and stores the result — no operators involved. It comes
that early so the first code you read is code that runs.

The rest of the guide expands each piece: programs, operators and the build-time boundary ([§4](#4-how-your-code-is-read-and-how-it-is-packaged)),
types and buffers ([§5](#5-data-types-and-buffers)), the datapath and its units ([§6](#6-the-datapath-model), [§8](#8-per-unit-primitive-contracts-what-is-legal-in-which-block)), compute intrinsics ([§7](#7-compute-intrinsics)), loops ([§9](#9-loops-build-time-unrolling-and-loop-lowering)),
predicates ([§10](#10-predicates-and-conditionals)), tiling ([§11](#11-tiling)), and fusion ([§12](#12-synchronization-and-fusion)).

**The `A` namespace at a glance** *(skim now; refer back as needed).* These are the language's
**programming abstractions** — half of what you write; the Python syntax carrying it is the other half
([§4.2](#42-the-parsed-form-and-what-it-accepts)). It falls into a few families:
- **Module & function decorators** — `@A.ir_module`, `@A.prim_func` (tag a class/function as an
  AxeleraScript module/function), plus `A.input` / `A.output` to name a module's operands and
  results.
- **Values & math** — data types and scalar/vector constructors (`A.int8`, `A.Broadcast`,
  `A.cast`, `A.IntImm`, …), and arithmetic ops (`A.Min`/`A.Max`, `A.min`/`A.max`).
- **Predicates & conditionals** ([§10](#10-predicates-and-conditionals)) — `A.if_then_else` selects a
  *value*; `A.If`/`A.Then`/`A.Else` and its shorthand `A.when` select a *code path*; `A.where`
  guards a region.
- **Loops** — `A.grid`, `A.serial`, `A.unroll` (device loops), plus `A.tiles` for tiling.
- **Buffers, scopes & streams** — `A.alloc_buffer`, memory scopes (`A.L1`, `A.L2`, `A.DDR`,
  `A.IMC`, …), and streams (`A.decl_stream`, `.push`/`.pop`, `A.drop`).
- **Datapaths** — `A.Datapath`, whose kind (`A.DP_MVMEXE` / `A.DP_DWPU`) is inferred
  ([§6](#6-the-datapath-model)), and `A.pseudo.LoadIMC`, a whole datapath of its own that makes a
  matrix resident ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)).
- **Blocks inside a datapath** — `A.IFD`, `A.pseudo.MatMul`, `A.DPU`, `A.ODR`
  ([§8.1](#81-the-block-contract-table)).
- **Movement blocks** — `A.CoreDMA`, `A.SysDMA`, written outside any datapath
  ([§8.1](#81-the-block-contract-table)).
- **Compute intrinsics** — `A.mvm_1x1`, `A.dpu.ops.*`, accumulators; depthwise goes through the
  operator library instead, since it has no `A.pseudo` block yet ([§6.3](#63-depthwise-operators)).
- **Metaprogramming & scheduling** — `A.script()`, `A.macro`, `A.sync`.

---

## Part II — Writing Programs

This part is how programs get written: what a complete one looks like, how the parser reads it and
how it is packaged as an operator ([§4](#4-how-your-code-is-read-and-how-it-is-packaged)), the data it moves ([§5](#5-data-types-and-buffers)), the datapath and what each unit
can compute ([§6](#6-the-datapath-model)–[§8](#8-per-unit-primitive-contracts-what-is-legal-in-which-block)), and the constructs that shape execution — loops ([§9](#9-loops-build-time-unrolling-and-loop-lowering)), predicates ([§10](#10-predicates-and-conditionals)),
tiling ([§11](#11-tiling)), synchronization and fusion ([§12](#12-synchronization-and-fusion)). It starts with the smallest program that runs.

A program is a sequence of datapaths, written out one after another. When a stretch of that code
is worth reusing, you package it as an **operator** and call it by name; wire operators together
and you have a **composition**, which is how a whole model is written. Choosing a level is about
how you organize code, not about what you can express:

| | what you write | where |
|---|---|---|
| 1 | a **program** — a module whose datapaths are written out | below |
| 2 | an **operator** — a named, reusable boundary around work of any size | [§4.3](#43-operators-as-the-usual-authoring-path-scriptop--arg) |
| 3 | a **composition** — operators wired together into a model | [§13](#13-high-level-model-authoring) |

**1. The smallest complete program.** A module (`@A.ir_module`) holding one entry function
(`@A.prim_func`) that scales its input by 2. No operator is involved: the datapath is written out,
and the stages it omits — the main compute stage and IFD1 — are bypassed for it ([§6.4](#64-bypass--writing-only-the-units-you-need)).

```python
SCALE = np.full((1, 64), 2.0, np.float32)                       # the multiplier, 2.0 in every lane

@A.ir_module
class Scale:
    @A.prim_func
    def main(inp: A.Buffer((8, 1), "float32x64", scope=A.DDR)):
        out = A.alloc_buffer((8, 1), "float32x64", scope=A.DDR)
        scale = A.alloc_constant(SCALE, dtype="float32x64", scope=A.DDR)
        for i in A.grid(8):                                     # one datapath per element
            with A.Datapath():                                  # kind inferred (§6)
                ifd = A.decl_stream("float32x64")
                dpu = A.decl_stream("float32x64")
                with A.IFD(num=0):                              # feed
                    ifd.push(inp[i, 0])
                with A.DPU():                                   # compute, in float18 (§7)
                    dpu.push(A.cast(ifd.pop(), "float18x64") * A.cast(scale[0], "float18x64"))
                with A.ODR():                                   # store
                    out[i, 0] = dpu.pop()
        A.output(out)                                           # name the module's result
```

Two things to carry forward. The body of a `@A.prim_func` **is** the parsed region, so no
`with A.script():` appears — that wrapper is for emitting script from inside an ordinary Python
method, which is the operator path ([§4.2](#42-the-parsed-form-and-what-it-accepts)). And DPU arithmetic needs **both operands as
`float18x64`**, with constants arriving from a buffer rather than as literals ([§7](#7-compute-intrinsics)).

**2. The same work, as an operator.** The datapath above is what `axol.atomic.mul` already does,
so the whole body collapses to one call:

```python
from axelera.compiler.alto import axol          # alto's operator library (§4.3)

@A.ir_module
class Scale:
    @A.prim_func
    def main(inp: A.Buffer((8, 1), "float32x64", scope=A.DDR)):
        A.output(axol.atomic.mul(inp, 2.0))
```

Same result as above.

**An operator is a reuse boundary, not a size.** The scale above is one datapath invocation, but a
convolution is at least two — making the weights resident in IMC comprises one invocation of its
own, run once, separate from the one that computes against them ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)) — and an operator that
unrolls a spatial axis emits one invocation per output position. Where you draw the boundary is a
design choice, not a limit of the language: one large fused kernel is as valid an operator as a
single elementwise step.

**Tiling and software pipelining cut across all three** ([§11](#11-tiling), [§12](#12-synchronization-and-fusion)); they are not a fourth way to
write code. An operator whose working set does not fit in **L1** — the only memory the IFD and ODR
reach — is tiled **inside its own implementation**: the compiler splits its outer loops against
memory by default ([§11](#11-tiling)), and the operator itself splits against a compute unit's own
capacity — the DPU register files, the IMC tile budget ([§14](#14-correctness-hazards--limits)). The same constructs apply **over a
composition**, tiling a graph of existing operators. Where to apply them is your choice, not the
language's.

That is the trade the rest of [Part II](#part-ii--writing-programs) is about — and none of it changes what the program above
already showed you about how a program executes.

**Reading order.** The sections that follow are ordered for reference, not for a first read. If the
datapath is new to you, take them in this order:

1. **[§4.1](#41-the-build-time--runtime-boundary)–[§4.2](#42-the-parsed-form-and-what-it-accepts)** — how the parser reads what you write
2. **[§5](#5-data-types-and-buffers)** — data types and buffers
3. **[§6](#6-the-datapath-model)–[§8](#8-per-unit-primitive-contracts-what-is-legal-in-which-block)** — the datapath, and what each unit can do
4. **[§4.3](#43-operators-as-the-usual-authoring-path-scriptop--arg)** — packaging the result as an operator

[§9](#9-loops-build-time-unrolling-and-loop-lowering)–[§12](#12-synchronization-and-fusion) are control and performance, and **[§14](#14-correctness-hazards--limits) lists the ways a program
can be silently wrong** — worth a skim before you debug anything, not after.

## 4. How your code is read, and how it is packaged

Four topics. Two are about **how the parser reads what you write**, and apply to any program,
operator or not: the boundary between the Python that runs at parse time and the AxeleraScript that
is executed at runtime ([§4.1](#41-the-build-time--runtime-boundary)), and the two forms a datapath can be
written in, with what the parsed one accepts ([§4.2](#42-the-parsed-form-and-what-it-accepts)). The
third is **packaging** — the `ScriptOp` authoring path ([§4.3](#43-operators-as-the-usual-authoring-path-scriptop--arg)).
The fourth is the **device loop constructs** ([§4.4](#44-device-loop-constructs)), the vocabulary the
sections after this one iterate with. Most kernels end up as **registered operators**, and most are
built inside a `with A.script():` block. Two kinds need no wrapper: a module written directly, as
in [Part II](#part-ii--writing-programs), and an operator that emits no datapath of its own — one composed from other operators
has nothing to parse.

> **Shapes are static.** Every extent and buffer dimension resolves to a compile-time constant
> before the program is executed — nothing stays dynamic at runtime. Most are fixed at
> parse time (metaprogramming computes them from Python variables / operator args); some stay
> **symbolic** longer and are resolved by the compiler during lowering — e.g. a tiling loop's
> tile factor, chosen within the bounds you set via `tile_min` / `tile_max` ([§11](#11-tiling)).

### 4.1 The build-time / runtime boundary

Parsed AxeleraScript lives in a **parsed region**: a `@A.prim_func` body, parsed in full, or a
`with A.script():` block inside an ordinary Python method — the operator path ([§4.3](#43-operators-as-the-usual-authoring-path-scriptop--arg)). [§4.2](#42-the-parsed-form-and-what-it-accepts) gives
the contract both obey. This section is about the boundary that runs *through* such a region.

Two modes coexist in the code you write:

- **Build-time Python (metaprogramming)** — runs once, at parse time, and never becomes
  executable: shape math, `int(...)` coercions of operator args, tile-count computation,
  Python `if` selecting a datapath variant, a **plain-Python** `for ... in range(...)` that
  *unrolls* ([§9.1](#91-the-two-for-rule)), and helper functions that build and **return** expression trees.
- **Executable code (pure parsed AxeleraScript)** — the AxeleraScript *statements* in the
  parsed region. Some are `A.*` constructs: `A.grid`/`A.serial`/`A.unroll` loops,
  `with A.Datapath(...)`, the unit blocks, `stream.push/pop`, `A.mvm_1x1`, `A.cast`, `A.where`,
  `A.If/Then/Else`. Others are plain Python syntax that the parser reads as executable code just
  the same — a **buffer store** (`out[oh, ow] = dpu_stream.pop()`), an accumulator update
  (`acc += …`), an `if`/`else` branch, index arithmetic. All of it emits IR that lowers to
  executable instructions.

**The boundary is not purely lexical** — it is *not* simply "outside the parsed region" versus
"inside it." Inside it, the parser interprets the
**AxeleraScript programming abstractions** — loops, unit blocks, `push`/`pop`, stores, conditionals, and
the `A.*` expression builders — to build IR. A call to **your own Python function** is *not*
part of those abstractions: the parser runs it as ordinary Python at parse time and inlines
whatever expression it **returns**. So metaprogramming can also run *inside* the parsed
region, as helper calls — as long as each helper **returns** the expression to use rather
than trying to emit statements.

**Example.** Build-time Python computes the values the parsed script is built from; the
parsed region describes the datapath. A helper called *inside* `with A.script():` still runs
at parse time:

```python
def _addr(i, j):                           # build-time helper: runs at parse time, RETURNS an expr
    return i * 64 + j

def implementation(self):
    n = int(self.rows)                     # build-time: arg coercion / shape math
    with A.script():                       # ─── parsed into IR below ───
        for i, j in A.grid(n, 64):         # device loop
            ...                            # datapath / stream setup (omitted)
            with A.IFD(num=0):             # device unit block
                ifd_stream.push(inp[_addr(i, j)])   # `_addr` runs as Python here (parse time);
                                                    #   only the index it RETURNS enters the IR
```

What that pushes is not a call to `_addr` — the call is gone by the time the IR exists, leaving
only the expression it returned:

```python
# what you write
with A.IFD(num=0):
    ifd_stream.push(inp[_addr(i, j)])

# what it parses to
with A.IFD(num=0):
    ifd_stream.push(inp[i * 64 + j])
```

`_addr` is defined outside the block but called from *inside* the `A.IFD` device block; it
still runs at parse time, and only its returned expression enters the IR. The same pattern
carries the bounds-checked/padded load in a real IFD block ([§8.3](#83-ifd--input-feed)).

**A helper must be *defined* outside the parsed region.** Only the call may sit inside it. A
`def` within the block is refused (a nested function is not part of those abstractions), so put
helpers at module scope, beside the method, or just above the `with A.script():` line, as
above. The error names the nesting and lists those three places; no decorator on the nested
`def` helps, including `@A.macro` ([§14](#14-correctness-hazards--limits)).

### 4.2 The parsed form, and what it accepts

There are **two ways to write a datapath**, and the constraints in the rest of this section
apply to the first only:

| | **parsed** — inside `with A.script():` | **imperative** — no script block |
|---|---|---|
| device loop | `for i in A.grid(N):` | `with A.grid(N) as i:` |
| stream | `s = A.decl_stream(dt)` | `with A.decl_stream(dt) as s:` |
| push | `s.push(v)` | either; also `A.stream_push(s, v)` |

**Write the parsed form.** It is what the rest of this guide uses, and it is what most of the
operator library is written in.

The **imperative** form is for an operator whose structure is itself computed in Python, since
ordinary Python statements interleave freely with it. Its body is executed rather than parsed, so
the refused-construct list below does not apply to it — but everything else in the guide does,
including the per-unit contracts of [§8](#8-per-unit-primitive-contracts-what-is-legal-in-which-block), which are checked on the emitted IR either way.

Both forms of parsed region ([§4.1](#41-the-build-time--runtime-boundary)) — the `@A.prim_func` body,
which is why [Part II](#part-ii--writing-programs)'s program needs no wrapper, and the
`with A.script():` block — obey one contract:
- Names from the enclosing Python scope are visible inside: module globals, and for a
  `with A.script():` block the function's local variables already bound above the `with` line
  (the build-time values computed there).
- The parsed region mixes the `A`-namespace constructs ([§3](#3-the-programming-model-in-one-page)) with a **subset of ordinary Python**,
  listed below. Anything outside that subset is refused at parse time, with an error naming the
  construct and pointing at your line.
- The body **must exist as real source in a `.py` file.** AxeleraScript is parsed from the
  function's *source text* (located by file and line number), not from its compiled/runtime
  form — so a body generated dynamically at runtime (e.g. built with `exec()`/`eval()`, or
  any function whose source cannot be retrieved) cannot be parsed.
- For the same reason, **`from __future__ import annotations` (PEP 563) at module scope**
  breaks a parameterized `prim_func` signature; the error points at `def main(`, not at the
  import ([§14](#14-correctness-hazards--limits)).

**What you may write.** Two categories, and the difference carries through the rest of the
guide: a **statement** is a step that is executed at runtime — it stands on its own line and *does*
something; an **expression** is a value used *inside* a statement, and never stands alone. That
is the whole of the helper-versus-macro distinction: a build-time helper **returns an
expression** ([§4.1](#41-the-build-time--runtime-boundary)), a macro **emits statements** (below).

These statements parse:

| Statement | Notes |
|---|---|
| `x = …`, `x: T = …`, `x += …` | assignment; `+=` is how an accumulator is updated ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)) |
| `for` | over a device loop or a build-time `range` — [§4.4](#44-device-loop-constructs) |
| `if` / `else` | a branch — **both arms are parsed**, see the caution below; [§10](#10-predicates-and-conditionals) covers when to prefer `A.if_then_else` or `A.where` |
| `with` | unit blocks, `A.Datapath`, `A.If`/`A.Then`/`A.Else`, `A.init`, `A.tiles` |
| a call on its own line | `stream.push(…)`, `A.sync(…)`, an operator call |
| `pass` | |

And these expressions: arithmetic (`+ - * //`), comparisons, `and` / `or`, unary minus, buffer
indexing and slicing (`buf[i]`, `buf[a:b]`), tuples, attribute access (`buf.shape`), a
conditional expression (`x if c else y`), starred unpacking (`A.grid(*shape)`), f-strings, and
list literals.

**What is refused.** Everything else, including: `while`, `assert`, `try`/`except`/`finally`,
`break`/`continue`, `raise`, `del`, `global`/`nonlocal`, `import`, `match`, `class`, a
**nested `def`** (which is why a build-time helper is defined *outside* the block — [§4.1](#41-the-build-time--runtime-boundary)),
`lambda`, comprehensions and generator expressions, assignment expressions (`:=`), and
**dict literals** (list literals do parse).

> **Caution — an `if` in a parsed region is not a Python `if`.** Python evaluates the condition
> and runs one branch. The parser does neither: it reads the whole statement, both branches
> included, and builds IR for both ([§10](#10-predicates-and-conditionals)). What becomes of the branch that cannot run depends on
> the condition:
>
> - **a build-time constant** — lowering deletes the dead branch, so it is never executed.
>   Branching on a flag your Python code computed is therefore free.
> - **a runtime value** — a loop variable, or anything else symbolic — there is nothing to delete,
>   and the branch is executed at runtime.
>
> Either way the parser has already read both branches, so **both must be valid AxeleraScript**,
> including one that can never execute. A name bound to Python `None` fails this way:
> `if x is not None:` does not protect the statements inside it, because the guard is not
> consulted until long after parsing ([§14](#14-correctness-hazards--limits)).

> **This describes what *parses*.** Parsing is not lowering: a construct can be accepted here
> and still be illegal in a particular position — for example, an intermediate assignment parses,
> but is refused inside an `A.IFD` block ([§8.3](#83-ifd--input-feed)). The per-unit contracts in [§8](#8-per-unit-primitive-contracts-what-is-legal-in-which-block) are the other half.

`@A.macro` is a named, reusable **block of AxeleraScript statements**: its body is parsed where
you call it, inlined at the call site with its own isolated scope (internal names stay private;
only the arguments you pass in cross the boundary). Unlike the build-time helper of [§4.1](#41-the-build-time--runtime-boundary) (which *returns
an expression*), a macro *emits script*. Like that helper, it must be **defined** outside the
parsed region — only the call goes inside.

### 4.3 Operators as the usual authoring path (`ScriptOp` + `Arg`)

Operators in **AXOL** — `alto`'s operator library, imported as
`from axelera.compiler.alto import axol` — come in three levels: **atomic** (a
single-datapath primitive), **core** (a larger op — either composed from atomic and other ops,
or built directly from several datapaths), and **aten** (a PyTorch-surface layer over core). This
section covers writing an **atomic** op — the minimal surface needed to start one.

> **Two things are called AXOL.** This guide always means `axelera.compiler.alto.axol` — the
> library of implemented operators you call from AxScript. The separately-packaged
> `axelera.axol` (`axelera-axol`) is the **graph compiler's op registry**: it declares each
> graph-level op with its tensor types, attributes, traits and PyTorch mapping, and points that
> op at a kernel — and those kernels are, for the most part, operators from this library.
> Writing AxScript directly, you call the library itself, so you are not limited to the ops that
> have a registry entry.

> **This section is about the wrapper, not the work.** What goes *inside* `implementation()` — units,
> streams, dtypes, intrinsics — is [§5](#5-data-types-and-buffers) through [§8](#8-per-unit-primitive-contracts-what-is-legal-in-which-block). If you have not read those yet, the example below
> will use a handful of constructs you have not met yet (`A.Datapath`, `A.decl_stream`, `A.IFD`, `A.DPU`,
> `A.ODR`); read it for the shape of an operator, and come back once the datapath is familiar.

To write and register a new operator you subclass `ScriptOp`: declare its arguments as `Arg`
fields, implement four methods, and register it with `register_op`. This section walks through
each, then shows a complete example.

**Declaring arguments — `Arg`.** An op subclasses `ScriptOp` and declares each parameter as a
class-level `Arg(...)`. Positional args bind in **declaration order**; `kw=True` makes one
**keyword-only** — it can only be passed by name at the call site, so `dpu_cast(x, core=0)` is
accepted and `dpu_cast(x, 0)` is not. That keeps optional parameters out of the positional order.

| `Arg(...)` field | Meaning |
|---|---|
| `expected_type` | allowed type(s): a **tensor** type (`Int8x64[tir.Buffer, "..."]`) for tensor operands, or a plain Python type (`int`, `tuple`, an enum, `... \| None`) for **scalar attributes** |
| `description` | human-readable purpose |
| `kw` | `True` → keyword-only (default `False` → positional) |
| `default` | value used when the arg is omitted (an arg with a `default` is optional) |
| `is_output` | `True` → an output buffer the op writes; auto-allocated when not supplied (see `infer_outputs`) |
| `variadic` / `trailing` | advanced: collect trailing positionals into a tuple / force the arg to the end of the list |

**Reading `Int8x64[tir.Buffer, "..."]`.** The class before the bracket constrains the **dtype**
— `Int8x64` admits only `int8x64`, and a union of classes admits each of them, as the example
below does — and the quoted string constrains the **shape**, with `"..."` meaning *any shape*.
Give a spec instead when the op needs one: `Float32x64[tir.Buffer, "1"]` is a single-element
buffer.

**Introspecting a tensor arg.** A bound tensor argument is a buffer handle: read `arg.shape`
(a static tuple), `arg.dtype` (e.g. `"int8x64"`), and `arg.scope()` — used in `validate` /
`infer_outputs` / `implementation`.

> **In `emulate()` these same attributes mean something else** — a NumPy dtype, and a shape carrying
> the lane axis; the example below is commented at the two lines that differ. The spelling is identical
> either way, so a wrong comparison is silently `False` rather than an error.

> **An operand may be a sub-region of a larger buffer** — a caller can pass a slice, and [§11](#11-tiling)'s
> `buf.tile(...)` produces one. `arg.shape` gives that region's **extents only**; *where* it starts
> is carried separately. So an op that re-materializes an operand as a plain buffer must alias the
> **whole parent** and re-apply the operand's own indices, never pair the region's shape with the
> parent's data. Getting this wrong reads and writes a valid but **wrong** region, silently ([§14](#14-correctness-hazards--limits)).

**The four methods** every op implements:
- `validate() -> None` — raise if the bound arguments are unsupported (clear message).
- `infer_outputs() -> dict[str, BufferInfo]` — return, per output arg, a
  `BufferInfo(shape, dtype, scope)`; the `str` key is the **name of the `Arg` field** it
  describes (`"output"` in the example below). Omitted / `None` outputs are auto-allocated from it.
  An output dtype is **inferred here, not taken as an argument** — from the inputs or from the compute
  unit's own result type. Which of those applies, and why an operator should not grow an `out_dtype`
  parameter, is a convention the shipped operators follow.
- `implementation() -> None` — emit the AxeleraScript body (inside `with A.script():`).
- `emulate(...)` — a NumPy reference implementation of the op's semantics.

**Registration.** `register_op(Cls, "axol.atomic.<name>", generation=<target-generation>,
atomic=True)`. `generation` names the target the op is registered for: pass the
`HardwareGeneration` value, or a list of them, for the targets you build for — the same value you
give `AltoConfig(generation=...)`. `atomic=True` marks a
single-datapath primitive (core/aten omit it — default `atomic=False`). The flag decides **when**
the op is expanded: the compiler decomposes composite ops first, folds constants over the result,
and expands the atomics only afterwards — so an atomic stays one opaque call while the ops built
from it are being taken apart.

**A complete atomic op** — a DPU cast (`int8x64` ⇄ `float*x64`), one datapath, IFD → DPU → ODR:

```python
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
from axelera.compiler.alto.compiler.dtypes import Float16x64, Float32x64, Int8x64
from axelera.compiler.alto.compiler.device import DatapathKind
from axelera.compiler.alto.axol.common import UnsupportedError
from axelera.compiler.alto.compiler.interpreter import emulation as emu
from axelera.config import HardwareGeneration
import axelera.compiler.alto.script as A
import numpy as np
from tvm import tir

class AtomicDpuCast(ScriptOp):
    CAST_TYPES = (Int8x64[tir.Buffer, "..."] | Float16x64[tir.Buffer, "..."]
                  | Float32x64[tir.Buffer, "..."])

    data   = Arg(CAST_TYPES, description="Input tensor.")
    output = Arg(CAST_TYPES | BufferInfo, default=None, is_output=True,
                 description="Output; None → auto-infer (matches data).")
    datapath_kind = Arg(DatapathKind | None, default=None, kw=True, description="Optional datapath kind.")
    core          = Arg(int | None,          default=None, kw=True, description="Optional core.")

    def validate(self) -> None:
        # dtype is a STRING in validate / infer_outputs / implementation — the lane count is part
        # of it ("int8x64"). Compare against strings here; see emulate() below, where they read differently.
        if self.data.dtype not in {"int8x64", "float16x64", "float32x64"}:
            raise UnsupportedError("Unsupported input dtype")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        return {"output": BufferInfo(shape=tuple(self.data.shape),
                                     dtype=self.data.dtype, scope=self.data.scope())}

    def implementation(self) -> None:
        with A.script():
            for loop_vars in A.grid(*self.data.shape, always_tuple=True):
                with A.Datapath(kind=self.datapath_kind, core=self.core):
                    ifd_stream = A.decl_stream(self.data.dtype)
                    dpu_stream = A.decl_stream(self.output.dtype)
                    with A.IFD(num=0):
                        ifd_stream.push(self.data[loop_vars])
                    with A.DPU():
                        dpu_stream.push(A.cast(ifd_stream.pop(), self.output.dtype))
                    with A.ODR():
                        self.output[loop_vars] = dpu_stream.pop()

    def emulate(self) -> None:
        # Here every operand is a NumPy array, so the SAME attributes read differently:
        #   .dtype  is a NumPy dtype  — np.int8, NOT "int8x64"  (note the contrast with validate)
        #   .shape  carries the lane axis — a buffer of (M, K_BLOCKS) int8x64 arrives as
        #           (M, K_BLOCKS, 64), so shape[-1] == 64 and negative axis indices shift by one
        # emu.rf18 rounds a float32 array to the DPU's native float18 precision. Apply it
        # whenever the op computes with the DPU's numerics — without it the reference is
        # too precise and the comparison fails (§7).
        x = emu.rf18(self.data.astype(np.float32))
        x = np.clip(np.round(x), -128, 127) if self.output.dtype == np.int8 else x
        np.copyto(self.output, x.astype(self.output.dtype))

dpu_cast = register_op(AtomicDpuCast, "axol.atomic.dpu_cast",
                       generation=<target-generation>, atomic=True)
```

`Arg` values are read at build time, but a scalar arg may arrive as a **TIR constant** rather
than a Python number: `range(n)` then raises, and `n // 64` silently stays a TIR expression.
Coerce with `int(...)` before doing Python arithmetic on one.

### 4.4 Device loop constructs

Inside a parsed region these produce **execution loops**, as
opposed to a Python `for … in range(…)`, which unrolls at build time ([§9.1](#91-the-two-for-rule)):

- **`A.serial(N)`** — a single sequential loop over `0 … N-1`, executed strictly in order
  (not reordered or parallelized). Unlike `A.unroll` (below), it is **not** unrolled into
  `N` separate copies **while it wraps at most one datapath**: the execution unit covers the
  whole range as one command with the loop extent encoded. A loop enclosing **two or more**
  datapaths has no single unit to carry it, so the compiler unrolls it after all — [§9.3](#93-two-or-more-datapaths-in-the-loop--the-loop-is-unrolled).
- **`A.grid(*extents)`** — an N-dimensional nested loop nest over the Cartesian product of
  `extents`, in **row-major order** (the last axis varies fastest). `for i, j in
  A.grid(H, W)` is the nest `for i in 0..H: for j in 0..W`.
  - **`always_tuple=True`** — delivers the loop variable as a single tuple even for one axis,
    so build-time code can slice it by position when the number of axes is computed
    dynamically:
    ```python
    for vars in A.grid(*shape, always_tuple=True):   # len(shape) not known when writing this
        spatial, ch = vars[:-1], vars[-1]            # slice by position — works for any axis count
    ```
- **`A.unroll(N)`** — a loop the **compiler** fully unrolls into `N` copies during lowering
  (device-side unroll; distinct from the parse-time Python `range` unroll).

`A.serial` and `A.unroll` also accept a two-argument `(start, stop)` form (like `range`): e.g.
`A.serial(1, N)` iterates `1 … N-1`. (`A.grid` axes are always extents, `0 … extent-1`.)

Loop **extents must be resolved to constants** by the time the loop is lowered. Loop
variables are integer indices; use them to index buffers (including strided slices, [§5](#5-data-types-and-buffers)).

A **zero extent is legal** and lowers to nothing. This matters because such a loop is easy to reach by
accident and hard to guard against: a `stride - 1` repeat count is zero at stride 1, and you cannot
avoid emitting it with a Python `if` inside the parsed region, since both arms are parsed anyway ([§4.2](#42-the-parsed-form-and-what-it-accepts)).
Write the loop and let it collapse. (From *outside* the parsed region the other route works too: a plain
Python `for _ in range(stride - 1)` that opens a script block per iteration emits nothing at all when
the count is zero — [§9.1](#91-the-two-for-rule). Inside the body, the device loop is the answer.)

---

## 5. Data types and buffers

Before the datapath itself, the shape of the data flowing through it. The fundamental vector unit
is the **PWORD** = **64 lanes**; most values are one PWORD (`x64`).
dtype strings are `"<base>x<lanes>"`, with the lane count always explicit. Two things carry a
different count: `int8x4096`, an IMC weight block (64 PWORDs, in the table below), and the packed
forms `float16x32` / `float32x16` that only `A.vtrsp` takes ([§8.4](#84-odr--output-store--drop)) — 32 or 16 wider elements
filling the same stream beat that holds 64 `int8` ones.

| dtype | Typical use |
|---|---|
| `int8x64` | activation/weight element; IFD stream; MatMul and depthwise input |
| `int8x4096` | one IMC weight block (64×64 int8) |
| `int32x64` | MatMul result stream |
| `float18x64` | DPU internal register |
| `float32x64` / `float16x64` | output / intermediate float |

`float18` is a reduced-precision float. Against `float32` it gives up **range** as
well as precision — see [§7](#7-compute-intrinsics) for what that costs you.

Scalar/vector helpers: `A.int8(v)`, `A.int32(v)`, `A.float32(v)`,
`A.Broadcast(scalar, lanes)`, `A.cast(value, "dtype")`. There is **no `float18` literal** — DPU
constants come from a buffer, not an immediate ([§7](#7-compute-intrinsics)).

`A.cast(value, "dtype")` converts a value to another dtype — it is how data crosses between the
streaming dtypes at unit boundaries (e.g. the MVM's accumulator → `float18` for the DPU →
`float32` out). It applies the target type's rounding/clipping: casting into `float18`
rounds to DPU precision ([§7](#7-compute-intrinsics)), and casting into `int8` rounds and clips to `[-128, 127]`. It is
an expression (returns a value).

> **PWORD is always 64.** The PWORD-packed dimension must be a multiple of 64; the buffer
> carries it as a trailing `/64` block dimension with an `x64` dtype (e.g. `(H, W, C // 64)`,
> `"int8x64"`). Concretely, a `56 × 56` feature map with `C = 128` channels is a buffer of shape
> `(56, 56, 2)` and dtype `"int8x64"` — the last dimension counts *blocks of 64 channels*
> (`128 / 64 = 2`), and the 64 channels within a block live in the dtype's lanes, not in the
> shape. [§6.2](#62-canonical-mvm-datapath--a-matrix-multiply) carries that same layer through a full convolution.
>
> **Which axis is packed: channels.** A PWORD holds 64 consecutive channels of one spatial
> position, so activations are stored channels-innermost — `(H, W, C // 64)`, the layout usually
> written **NHWC**. That is what makes one push a complete vector for the units: a MatMul's 64 lanes
> are 64 terms of the reduction ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)), and a depthwise op's are 64 channels computed side by side.
> IFD and ODR addressing assumes the same ordering. A model given in NCHW has to be
> permuted on the way in ([§13](#13-high-level-model-authoring)).

**Indexing.** A buffer access `buf[i, j, k]` reads/writes one element. An index may also be
a **strided slice** `A.Ramp(base, step, lanes)` — `lanes` elements starting at
`base`, spaced by `step` (e.g. `A.Ramp(row*64, 1, 64)` selects a 64-wide contiguous block).
A contiguous ramp can also be written with Python slice sugar: `buf[base : base+lanes]` is the
same as `A.Ramp(base, 1, lanes)` (e.g. `inp[64:128]`).
Addresses must be **affine** (linear) in the loop variables in the movement units (IFD / ODR
/ DMA, [§8](#8-per-unit-primitive-contracts-what-is-legal-in-which-block)).

### 5.1 Memory (buffer) scopes — `BufferScope`

| Constant | Use |
|---|---|
| `A.L1` | L1 memory, local to one AI core |
| `A.L2` | L2 memory, shared by all cores |
| `A.DDR` | external memory, shared |
| `A.IMC` | the store the resident matrix lives in, local to one AI core |
| `A.*_WEAK` (`A.L1_WEAK`, …) | "weak" — placement passes may move it |
| `A.L1_<n>`, `A.IMC_<n>` | pinned to core `n` (also `_WEAK` forms) |

Prefer the `*_WEAK` forms unless you must pin: a weak scope is one the placement passes may
move — elsewhere in the same memory, or to a different memory — which is what lets them fit your buffers
around everything else. Pin only when the placement itself is part of the design. `buf.scope()`
returns a buffer's scope string — reuse it to match an input's scope.

A scope says **where** data is accessed from, not **how much** of it is resident. Even a pinned
scope makes no promise that the whole buffer sits there at once: when a buffer does not fit, the
compiler tiles the loop that accesses it and streams tiles through with DMA ([§11](#11-tiling)), so a buffer
declared `A.L1` can be larger than L1 and still compile — provided there is a loop to tile. Do not
infer a size limit from a scope; a buffer's extent and the memory's capacity ([§14](#14-correctness-hazards--limits)) are separate
concerns. Those capacities do matter when you choose a tile size — that is the one decision where
whether the data fits is the question.

### 5.2 Allocation

```python
buf   = A.alloc_buffer(shape, dtype, scope=A.L1_WEAK)    # weak: let placement move it (§5.1)
const = A.alloc_constant([0.0]*64, "float32x64", scope=inp.scope())

imc   = A.alloc_imc((K_BLOCKS, N_BLOCKS), dtype="int8x4096", core=0)  # logical resident matrix

acc   = A.mvm_accumulator()                              # (1,), MVM_ACCUMULATOR
A.reset_accumulator(acc)                                 # zeroes every lane
```

`A.alloc_imc(...)` allocates a **logical** IMC buffer sized to the matrix you make resident — you
pick the `(K_BLOCKS, N_BLOCKS)` shape in 64×64 tiles (and optional core), and the compiler places
it.

`A.mvm_accumulator()` is likewise not general memory: it is a one-element handle to the MVM's
own accumulator register. Being a register, it is read and written
**without an index**, the same as `A.var` below: `A.reset_accumulator(acc)` zeroes it, and MAC
results accumulate into `acc` inside a MatMul block ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)). `acc[0]` is a parse error.

**Register files.** The DPU works out of register files of `float18x64`, its working dtype:

- `A.array(dtype, size=N)` — an `N`-register file, **indexed `reg[i]`**. The **scope is inferred
  from the dtype**: `float18x64` → DPU registers. `size` has a large
  default; pass the count you actually need. `N` must be a build-time positive integer (its
  upper bound is a parser limit, not a register count), or a device-loop expression the register allocator sizes
  later; anything else is refused where you wrote it.
- `A.var(dtype)` — a single register, **accessed directly (not indexed)**: assign to `v` and read `v`.
  (`A.array` above is the indexed form; indexing an `A.var` fails with *"Cannot index v with
  [0]: it is a var (a.k.a. single element buffer)"* on a read, and *"Cannot store into v at 0"* on a
  write; both name the register and the form to write instead.)

> **A value used by a *later statement* must be a declared register** — an `A.var` or an `A.array`.
> Binding a plain Python name and reading it in a later statement is **not supported** and fails at
> compile time ([§14](#14-correctness-hazards--limits)). Within one statement it is free: an expression assembled inline, or returned by a
> Python helper and passed straight into the statement that consumes it ([§8.3](#83-ifd--input-feed)), never crosses a
> statement boundary.
Registers are storage the DPU owns, so a file may only be read or written from a DPU block.
Declaring on a `Datapath` lets several DPU blocks share one file; it does not open it to the
other blocks.

> **Where you may declare one.** A register file must be declared inside a DPU block or on a
> `Datapath` — one declared inside any other block is refused, naming the block you are in. The general buffer constructors refuse
> a register scope outright: `A.alloc_buffer(..., scope=A.DPU_REGISTERS)` tells you to use
> `A.array` instead, because it checks neither the register count nor the block the file is
> declared in.

> **How you may index one.** One register at a time, by a whole number. A slice (`reg[0:4]`), an
> `A.Ramp`, and a non-integer index are each refused: a register file is not a vector you can take
> a run out of. A literal index is checked against the file's size, so `reg[12]` on a file
> declared `size=8` is an error at your line rather than a lowering failure later. An index the
> compiler cannot evaluate, such as a device-loop variable, is left to the register allocator.

See [§8.5](#85-dpu-register-programs) for how a DPU block reads and writes these registers.

---

## 6. The datapath model

This is the core of the language: the pipeline [§2](#2-the-execution-model-that-the-language-targets) described, as you actually write it. A datapath
is a `with A.Datapath(kind=..., core=...):` region containing per-unit `with`-blocks. The concept — a concurrent pipeline of stream units wired by FIFOs, in one of
the two kinds — is introduced in [§2](#2-the-execution-model-that-the-language-targets); this section covers it in detail: streams ([§6.1](#61-streams)), the
MVM and depthwise sections ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)–[§6.3](#63-depthwise-operators)), and how omitted units are bypassed ([§6.4](#64-bypass--writing-only-the-units-you-need)).

Both `kind` and `core` are **optional**: omit `kind` and the compiler infers it — a datapath
holding a `pseudo.MatMul` is `DP_MVMEXE`, and every other one is `DP_DWPU` — and omit `core` to let
it choose placement. The inferred kind decides which datapaths can run in parallel ([§2](#2-the-execution-model-that-the-language-targets)).
`A.pseudo.LoadIMC` needs no `kind` at all: it is always a datapath of its own.
Pass them only to pin a specific datapath kind or core.

To place **a whole region** on one core rather than a single datapath, wrap it in
`with A.AICore(core=n):` — buffers and operators inside it are allocated and run there. Work
sharded across cores this way is independent, so ordering between the shards needs an explicit
`A.barrier(*cores)` ([§12](#12-synchronization-and-fusion)); nothing infers it.

**Ordering & visibility.** Within a datapath, ordering is carried by the **stream FIFOs**:
each stream delivers elements in push order, and a consumer stalls until its input arrives —
so units are ordered by stream *data dependencies*, not by textual/program order (several
blocks of the same unit still run in the order written, [§9.2](#92-one-datapath-in-the-loop--the-loop-is-sunk-into-the-datapath)). The execution
model provides no *intrinsic* shared-memory ordering; where datapaths communicate through a shared
buffer rather than a stream, the order is program order — by default each launch waits for the one
before it to finish ([§12](#12-synchronization-and-fusion)). Ordering across cores must be stated explicitly with
`A.barrier(*cores)` / `A.sync` ([§12](#12-synchronization-and-fusion)).

> **The IMC is ordered by program order.** A weight load always forms a datapath invocation of its
> own, so it and the matmul that reads it are two invocations in program order — correctly ordered even when the
> matrix is replaced per iteration. You owe no `A.sync`. Note what that ordering is: each load is
> *moved* next to the compute that consumes it, and moved rather than copied. Every matrix's first
> use is therefore correct, while a reuse after another matrix has been used reads the wrong
> weights — no load was placed in front of it ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)).

**A unit can appear more than once in a datapath.** You may write several blocks of the same
unit, and they run as several commands on that unit, one per block.

### 6.1 Streams

```python
ifd_stream = A.decl_stream("int8x64")   # device scope inferred from the unit that uses it
v = ifd_stream.pop()          # consume one element
ifd_stream.push(value)        # produce one element
A.drop(some_stream.pop())     # consume-and-discard (valid only in ODR)
```

`A.drop(stream.pop())` consumes one stream element and throws it away — the pop still happens
(so the stream stays in order), but the value is not stored or used. It exists for the ODR
drop pattern ([§8.4](#84-odr--output-store--drop)): when the pipeline produces more positions than you want written out, the
surplus ones are dropped instead of stored.

You normally omit `scope`: the compiler assigns each stream's device scope from the unit that
pushes/pops it (a stream fed by `A.IFD(num=0)` becomes `IFD_0_STREAM`, a DPU output becomes
`DPU_STREAM`, …); pass `scope=...` only to pin one. These stream/device scopes name the
**stream or register a value binds to** — a different axis from the [§5.1](#51-memory-buffer-scopes--bufferscope) **memory
scopes** (`A.L1`/`A.L2`/`A.DDR`/`A.IMC`), which name where a *buffer* lives:
`IFD_0_STREAM`, `IFD_1_STREAM`, `MVM_STREAM`, `DPU_STREAM`, `ODR_STREAM` (and register scopes
`MVM_ACCUMULATOR`, `DPU_REGISTERS`).

A stream's **dtype must suit the unit that consumes it**: MatMul and the depthwise ops require
`int8x64` inputs (and produce a wider integer result); the DPU and the mover units
(IFD / ODR / DMA) also carry `float16` / `float32` streams, with DPU math done at `float18` ([§7](#7-compute-intrinsics)).
The dtype you pass to `A.decl_stream` names what the stream carries; a compute unit pushing its
own wider result into it is normal — [§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)'s MatMul pushes its accumulator into a stream
declared `int32x64`, the width of its result.

### 6.2 Canonical MVM datapath — a matrix multiply

The canonical **IFD → MatMul → DPU → ODR** flow, in three parts: the shape contract every MVM
operator obeys, that contract as a plain matrix multiply, then the same datapath as a convolution —
three lines changed.

#### The shape contract

**The MatMul does exactly one thing: `out = x @ W`, with `W` resident in the IMC.** A convolution is
not a different datapath — it is this matmul with the sliding window *flattened into the reduction*:
each output pixel's `KH · KW · C_in` window is one long vector `x`, and `W` is
`(KH · KW · C_in) × C_out`. There is one contract to learn; the convolution below only re-labels
its axes.

**The same matrix, three shapes.** One weight matrix appears under three names — the shape you
think in, the shape you store, and the shape the IMC holds — shown symbolically and for the worked
instance below (`K = 128`, `N = 64`):

| | name | shape | dtype | one element is |
|---|---|---|---|---|
| logical | `W` | `(K, N)` → `(128, 64)` | `int8` | one weight |
| in memory | `weights` | `(K, N_BLOCKS)` → `(128, 1)` | `int8x64` | row `k`; 64 output columns in the lanes |
| in IMC | `w_imc` | `(K_BLOCKS, N_BLOCKS)` → `(2, 1)` | `int8x4096` | one 64×64 tile |

`K_BLOCKS = K // 64`, `N_BLOCKS = N // 64`. The arithmetic is `out[m, n] = Σ_k x[m, k] · W[k, n]`.

**Which 64 is which.** 64 is the PWORD ([§2](#2-the-execution-model-that-the-language-targets)), so it lands on every axis — counting something
different each time:

| where you see 64 | what it counts |
|---|---|
| `int8x64` lanes of `inp` | 64 input channels — one PWORD of the reduction |
| `int8x64` lanes of `weights` | 64 **output** columns |
| a tile's 64 rows | 64 reduction terms `k` |
| a tile's 64 columns | 64 output columns `n` |
| `int8x4096` | 64 × 64 = one tile |

#### As a matrix multiply

```python
# inp: (M, K_BLOCKS) int8x64   weights: (K, N_BLOCKS) int8x64 constant   out: (M, N_BLOCKS) float32x64
# K_BLOCKS = K // 64 — one PWORD per block (§5)
w_imc = A.alloc_imc((K_BLOCKS, N_BLOCKS))                       # resident weights; int8x4096 by default
for kb, nb in A.grid(K_BLOCKS, N_BLOCKS):
    with A.pseudo.LoadIMC():                                    # fills IMC once, one block per iteration
        w_imc[kb, nb] = weights[kb * 64 : kb * 64 + 64, nb]     # 64 rows -> one IMC block (see below)

for m, nb in A.grid(M, N_BLOCKS):                               # one output block per datapath
    with A.Datapath(kind=A.DP_MVMEXE):
        ifd_stream = A.decl_stream("int8x64")
        mvm_stream = A.decl_stream("int32x64")
        dpu_stream = A.decl_stream("float32x64")

        for kb in A.serial(K_BLOCKS):                            # IFD: stream the reduction
            with A.IFD(num=0):
                ifd_stream.push(inp[m, kb])

        with A.pseudo.MatMul():                                  # MAC against the resident weights
            acc = A.mvm_accumulator()
            A.reset_accumulator(acc)
            for kb in A.serial(K_BLOCKS):
                acc += A.mvm_1x1(ifd_stream.pop(), w_imc[kb, nb])
            mvm_stream.push(acc)

        with A.DPU():                                            # cast to the output dtype
            dpu_stream.push(A.cast(mvm_stream.pop(), "float32x64"))

        with A.ODR():                                            # store the result block
            out[m, nb] = dpu_stream.pop()
```

**The three statements are the whole contract.** A MatMul block holds a reset, a reduction loop
over `acc += A.mvm_1x1(...)`, and a push of `acc`. Nothing else belongs in it, including a second
accumulate, a read of `acc` mid-reduction, or a second register. Each of these is refused with a
message that names your own register and the form to write instead.

**A worked instance.** `out = inp @ W`, with `inp` `4 × 128` and `W` `128 × 64`. In the program
each PWORD-packed dimension counts blocks of 64:

| | logical | in the program |
|---|---|---|
| `inp` | `4 × 128` | `(4, 2)` `int8x64` — 4 rows × 2 blocks of 64 reduction elements |
| `w_imc` | `128 × 64` | `(2, 1)` `int8x4096` — 2 × 1 tiles, each 64 reduction rows × 64 output columns |
| `out` | `4 × 64` | `(4, 1)` `float32x64` — 4 rows × 1 block of 64 output columns |

With those numbers, the datapath above computes:

```text
for m in 0..3:                        # 4 output blocks — one per row, since N_BLOCKS = 1
    acc = 0                           # the MVM accumulator
    for kb in 0..1:                   # K_BLOCKS = 2 — the IFD pushes 2, the MatMul pops 2
        acc += inp[m, kb] @ w_imc[kb, 0]
    out[m, 0] = acc                   # cast to float32x64 in the DPU
```

> **The IMC's layout, and why the load names 64 rows.** The IMC holds tiles indexed
> `(K_BLOCKS, N_BLOCKS)`, each one 64 reduction rows × 64 output columns — `(K_BLOCKS, N_BLOCKS,
> K_PWORD, N_PWORD)` with the tile unpacked. Your weight buffer does **not** have to be in that
> shape: it stays as you already have it, `(K, N_BLOCKS)` `int8x64`, and the slice
> `weights[kb * 64 : kb * 64 + 64, nb]` maps 64 of its rows onto one tile. That is the point of
> writing the load as a slice — it programs the IMC without repacking the weights.
>
> What the slice must get right is the **row extent**. The load takes *how many rows it writes*
> from the slice's extent, while it streams whatever data the slice names, so the two must agree —
> hence 64 rows, one tile. Naming a **single** pre-packed `int8x4096` value instead
> (`w_imc[kb, nb] = weights[kb, nb]`) declares a one-row load and then streams a whole block into
> it, which fails at run time with
> `mvmprg: Weights stream has not been fully consumed`.
>
> The operator library wraps exactly this loop as `atomic.load_imc` ([§11.2](#112-directing-it--atiles)), so in practice you
> call that rather than writing it.

**The two operands of `A.mvm_1x1` mean different things by their lanes** — the streamed vector's 64
lanes are 64 *reduction* terms, the tile's are the 64 *output* columns. Swapping them is silent. `A.alloc_imc((K_BLOCKS, N_BLOCKS))` is your matrix measured in tiles, and
`w_imc[kb, nb]` selects one.

> **What you owe the IMC is the tile budget; placement is automatic.** You give `A.alloc_imc` the
> matrix in tiles, and the one thing you must respect is the cap: **one residency holds a bounded
> number of `K_BLOCKS · N_BLOCKS` tiles** ([§14](#14-correctness-hazards--limits)).
>
> Where each tile goes below that is chosen for you, and a matrix that does not fit one residency
> is tiled along K ([§14](#14-correctness-hazards--limits)).
>
> **A second resident matrix on the same core replaces the first, not beside it.** The cap above
> is what *one* `A.alloc_imc` holds on one core, not a budget several share, and loading a new
> matrix ends the previous one's residency even when the previous one still fits under the cap —
> reload a matrix immediately before each reuse if another has been loaded since
> ([§14](#14-correctness-hazards--limits)). Buffers on different cores do not collide
> ([§5.1](#51-memory-buffer-scopes--bufferscope)).

#### The same datapath as a convolution

A convolution is the matmul above with the window flattened into the reduction. Take a real layer:

```text
① the layer      3x3 conv, 56x56 input, C_in = 128, C_out = 256
                    │  flatten the window into the reduction
② one matmul     K = KH·KW·C_in = 3·3·128 = 1152        N = C_out = 256
                 out[oh, ow, n] = Σ_k window(oh, ow)[k] · W[k, n]
                    │  cut into 64x64 tiles
③ what you write A.alloc_imc((1152 // 64, 256 // 64)) = (18, 4)  ->  72 tiles
                    │  placement (compiler)
④ residency      72 tiles — within one residency.
```

Only three lines differ from the matmul — the `LoadIMC` loop, the MatMul block, the DPU and the ODR
are unchanged:

```python
for oh, ow, nb in A.grid(OH, OW, N_BLOCKS):              # 1. one datapath per output pixel
    with A.Datapath(kind=A.DP_MVMEXE):
        ...                                              #    stream declarations, unchanged

        for kh, kw, cib in A.grid(KH, KW, C_IN_BLOCKS):  # 2. stream the window, not a row
            with A.IFD(num=0):
                ifd_stream.push(inp[oh + kh, ow + kw, cib])

        with A.pseudo.MatMul():                          #    the block itself is unchanged
            acc = A.mvm_accumulator()
            A.reset_accumulator(acc)
            for kb in A.serial(K_BLOCKS):                # 3. K_BLOCKS = KH * KW * C_in / 64
                acc += A.mvm_1x1(ifd_stream.pop(), w_imc[kb, nb])
            mvm_stream.push(acc)

        ...                                              #    DPU and ODR, unchanged
```

The two loops are **siblings inside the datapath**, not nested: one wraps the IFD block, the other lives
inside the MatMul block. Their **counts must match**: the IFD pushes
`KH · KW · C_IN_BLOCKS = 3 · 3 · 2 = 18` PWORDs and the MatMul pops `K_BLOCKS = 18`. That equality is
the contract.

**The dimensions, side by side.** For the layer above:

| the network's | in these shapes |
|---|---|
| input `(H, W, C_in)` = `(56, 56, 128)` | `inp` as `(56, 56, 2)` `int8x64` — `C_in/64 = 2` blocks |
| kernel `(C_out, KH, KW, C_in)` = `(256, 3, 3, 128)` | `weights` as `(1152, 4)` `int8x64` — row `k`, lane `c_out` |
| output `(OH, OW, C_out)` = `(54, 54, 256)` | `out` as `(54, 54, 4)` `float32x64`, `OH = H − KH + 1` |

**Which tile holds what.** The packing is the matmul's: 64 consecutive rows of `weights` become one
`int8x4096` tile, 64 reduction rows × 64 output columns. What changes is what a row *means*:
flattening the window makes row `k` a (tap, input-channel) pair — a **tap** being one position of
the kernel window, so the 3×3 kernel here has 9:

```text
k = (kh · KW + kw) · C_in + ci
```

So one tile holds 64 input channels of a single tap. With `C_in = 128` that is 2 tiles per tap, and
`K_BLOCKS = 18 = 9 taps × 2 blocks`; only when `C_in = 64` does a tile coincide with a whole tap.
That formula is **the order the code above pushes in**, not a fixed rule — see the
positional-pairing note below.

Note the two loops have **different shapes**: the IFD nests over `(kh, kw, cib)` while the MatMul
keeps one flat loop. Only the counts have to agree — the contract stated above; loop depth and
individual trip counts are free. Here the nesting is what keeps the IFD's address affine (a flat
`A.serial(K_BLOCKS)` would need `kb // KW`, which is not affine and is refused —
[§14](#14-correctness-hazards--limits)), but that follows from this address expression, not
from a rule about loop shapes. The two are paired only by stream order.

**The reduction is paired positionally**: the *n*-th value popped meets tile `w_imc[n, nb]`. So the order
you push the window in is your choice — window-major, channel-major, anything — as long as the weights
were packed in the same order. Nothing in the language fixes it, and nothing checks it, so a mismatch is
silently wrong.

`A.IFD(num=0)` names the main feeder, the one both examples use; `num` is optional and defaults
to `0`. Feeder routing is [§8.2](#82-how-the-blocks-interact--topology-and-ifd-feeder-routing).

#### Adding an epilogue

A real layer rarely stops at the product: a bias, a scale, an activation. Those belong in the **DPU**
block, which is already there to convert the result. A per-output-column bias held as a constant is
one statement — note that two `A.cast`es plus a single `+` is legal, because casts do not count
toward the one-operation-per-statement rule ([§8.5](#85-dpu-register-programs)):

```python
with A.DPU():
    dpu_stream.push(A.cast(A.cast(mvm_stream.pop(), "float18x64")
                           + A.cast(bias[nb], "float18x64"), "float32x64"))
```

`bias` is an ordinary constant buffer and **`nb` is the datapath's own loop variable** — a DPU block may
index a buffer at a device-loop index, not only at a literal one. Beyond one or two such terms, load the
constants once into registers with `A.init` instead ([§8.5](#85-dpu-register-programs)).

> **`A.pseudo.MatMul` and `A.pseudo.LoadIMC` are the whole interface to the matrix multiply.**
> `MatMul` is a block inside a `DP_MVMEXE` datapath. `LoadIMC` is a datapath of its own, written
> outside any `A.Datapath`, and the loop over IMC blocks makes the matrix resident once, before the
> multiplies that read it — a residency lasts until you load a different matrix. What surrounds the
> two — streaming the weights into the IMC, and summing the pieces of a long reduction within one
> residency — the compiler adds during lowering; you never write it.

### 6.3 Depthwise operators

Depthwise convolution and pooling run on the `DP_DWPU` datapath. The `DP_DWPU` datapath has no
`A.pseudo` compute block yet — one is planned — so these operators are written through the
operator library. Each call makes the filter resident, tiles the input, and covers every channel
block itself:

- `axol.core.dw_convolution_2d(data, weight, stride=(sh, sw), dilation=(dh, dw), padding=(0, 0))`
  — `data` is `(H, W, C_BLOCKS)` `int8x64`, `weight` is `(C_BLOCKS, KH, KW, 1)` `int8x64`, one
  filter per channel in the lanes;
- `axol.core.dw_convolution_1d(data, weight, stride=s, dilation=d, padding=0)` — the same over
  `(L, C_BLOCKS)`, with `weight` `(C_BLOCKS, K, 1)`;
- `axol.core.dwpu_pool(input, pool_type=..., kernel=..., stride=..., dilation=..., padding=0)` —
  `pool_type` is one of `axol.core.DwpuPoolType.MAX` / `AVERAGE` / `SUM` / `MIN`, and `kernel`'s
  length picks 1-D or 2-D pooling; `ceil_mode=True` rounds the output size up, and
  `divisor_override` sets the `AVERAGE` divisor in place of the window area.

A 3×3 depthwise convolution over two channel blocks:

```python
from axelera.compiler.alto import axol
import axelera.compiler.alto.script as A

KH, KW = 3, 3
C_BLOCKS = 2                                 # 128 channels
OUT_H, OUT_W = 8, 16
IN_H, IN_W = OUT_H + KH - 1, OUT_W + KW - 1  # the input the window reads

@A.ir_module
class DepthwiseConv3x3:
    @A.prim_func
    def main(
        inp: A.Buffer((IN_H, IN_W, C_BLOCKS), "int8x64", scope=A.DDR),
        flt: A.Buffer((C_BLOCKS, KH, KW, 1), "int8x64", scope=A.DDR),   # per-channel taps
    ):
        out = axol.core.dw_convolution_2d(inp, flt, stride=(1, 1), dilation=(1, 1),
                                          padding=(0, 0))
        A.output(out)
```

The output is `float32x64`, in the input's scope. Padding must be zero in every one of these
ops, and a window too large for the op to tile is rejected where you write it
([§14](#14-correctness-hazards--limits)).

### 6.4 Bypass — writing only the units you need

A datapath's chain of stages ([§2](#2-the-execution-model-that-the-language-targets)) is filled in automatically: **any stage you omit is
bypassed**, so a DPU-only op can write just `IFD → DPU → ODR`. The optional `IFD(num=1)` (the DPU
side-input feeder, [§8.2](#82-how-the-blocks-interact--topology-and-ifd-feeder-routing)) is bypassed simply by omitting it. This is why the skeletons above
name only the stages they use.

The compiler does all this automatically; `A.bypass()` explicitly marks the enclosing stream
unit bypassed, but is rarely needed.

Taken to its limit, a datapath may contain **no compute at all** — the ODR pops what the IFD pushed:

```python
with A.Datapath():                       # every compute unit bypassed
    s = A.decl_stream("float32x64")
    with A.IFD(num=0):
        s.push(inp[i, 0])
    with A.ODR():
        out[i, 0] = s.pop()
```

This is the right shape for pure rearrangement — copies, padding, replication, transposes. Prefer it
over routing values through the DPU when you have no arithmetic to do: DPU registers are `float18x64`
([§7](#7-compute-intrinsics)), so a value that passes through them is rounded, and a movement operator would stop being exact.

---

## 7. Compute intrinsics

[§6](#6-the-datapath-model) wired the units together; this section is what each one can actually compute. Each compute unit
performs a small, fixed set of intrinsics — called inside that unit's block — grouped below by unit
(MatMul, depthwise, DPU).

**MatMul** — `A.mvm_1x1(ifd_vec, imc_block)` : `int8x64 × int8x4096 → MVM accumulator`, a
matrix-vector product against **one** IMC tile. The `1x1` counts *tiles* — one reduction block
by one output block — not kernel pixels, which is why a `K_BLOCKS`-long reduction is a loop of
`K_BLOCKS` calls ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)). It is the **only** MatMul intrinsic — matmul, convolution, and
linear layers all reduce to a loop of `mvm_1x1` against resident IMC weights. Accumulate
across the reduction with `A.mvm_accumulator()` / `A.reset_accumulator(acc)` then
`acc += A.mvm_1x1(...)`, and the weights stay resident in IMC (loaded once,
[§6](#6-the-datapath-model)).

**Depthwise** — written through the operator library rather than through intrinsics:
`axol.core.dw_convolution_2d`, `dw_convolution_1d` and `dwpu_pool` ([§6.3](#63-depthwise-operators)).

**DPU** — `A.dpu.ops.*`: `lut`, `prelu`, `dpu_reduce`, `dpu_fma`,
`add_reduce`/`max_reduce`/`min_reduce`, and `fma(a, b, c)`. A softmax is composed from these: a
running `A.max`, `max_reduce`, an exponential `lut`, `add_reduce`, then a `1/x` `lut`.

**`A.dpu.ops.lut(x, table)` is a piecewise-linear approximation**, not a table of output values.
`table` is one PWORD of fitted parameters, which you build rather than write out:

```python
from axelera.compiler.alto.axol.utils.lut_chain import (
    compute_chained_lut_params, pack_lut_params_to_pword)

params = compute_chained_lut_params(x_min=0.0, x_max=8.0, n_luts=1, func=np.exp)
table = A.alloc_constant(pack_lut_params_to_pword(params[0]), dtype="float32x64", scope=A.DDR)
```

Inside a DPU block the table is an operand like any other, cast to `float18x64`:

```python
x = A.dpu.ops.lut(x, A.cast(table[0], "float18x64"))
```

Evaluation picks the piece `x` falls in and returns **`slope · x + offset`**, with float18 rounding
at each step and on the parameters themselves. Two things to get right when you fit a table:

- **Fit the domain to the data the operator sees**, and clamp `x` before the call if it can fall
  outside. A table is only defined over the range it was fitted for.
- **One table holds a bounded number of pieces.** For a finer approximation raise `n_luts`: the
  tables are *chained*, and `compute_chained_lut_params` splits the fit across them.

**Two rules govern every DPU arithmetic statement:**
- **Cast both operands to `float18x64` yourself.** A mixed pair is not reliably refused: it is
  usually promoted to `float32x64` at parse time without a message, and a `float18x64` value
  against a direct `float32x64` buffer read fails late with
  `mismatched types. float32x64 vs. float18x64`, without a source line.
- **A constant operand comes from a buffer.** There is no float18 literal, so allocate the value
  with `A.alloc_constant`, then either read it directly from that buffer inside the block — as the
  scale-by-2 program at the start of [Part II](#part-ii--writing-programs) does — or stream it in on the IFD1 side-input
  feeder ([§8.5](#85-dpu-register-programs)).

A statement also carries **at most one** arithmetic operation — see [§8.5](#85-dpu-register-programs).

The reduce ops `add_reduce` / `max_reduce` / `min_reduce(x)` (thin wrappers over
`dpu_reduce(x, op)`) reduce a single 64-lane register **across its lanes**, returning the
sum / max / min broadcast back across all 64 lanes. To reduce a dimension wider than one PWORD,
first combine the blocks elementwise (a serial loop of `+` / `A.max`) into one register, then
apply the reduce. Integer data reduces the same way once cast to `float18x64` — see the precision
note below for when that stays exact.

**Numerical precision.** Compute runs in fixed native formats — MatMul and depthwise
accumulate at a wider integer width than their inputs, the DPU operates in
`float18` ([§5](#5-data-types-and-buffers)), and transcendentals (`exp`, `1/x`, …) are evaluated
through LUT approximations. Results therefore reflect that precision and are **not**
bit-identical to a full-precision (`float32`/`float64`) reference; compare against a
reference with tolerances, not exact equality.

> **`float18` also has less range than `float32`, not just less precision.** A value that is
> comfortable in `float32` can overflow or flush to zero the moment it is cast for DPU
> arithmetic, and the result is wrong numbers rather than an error. If your operator handles
> large magnitudes — an unnormalized exponential, a sum of many large terms — scale before the
> DPU sees them.
>
> **Exact integers stay exact only in the accumulator.** An `int8` result is safe through the DPU, its
> whole range surviving the round trip, while a large integer accumulation destined for a
> `float32` output is **not**: it loses low-order bits on the way, and so does an integer reduction
> written in the DPU. That is a real characteristic of results that reach memory through the DPU,
> and not a reason to avoid it — just something to account for when the values can grow large.

---

## 8. Per-unit primitive contracts (what is legal in which block)

[§6](#6-the-datapath-model) showed the datapath in outline and [§7](#7-compute-intrinsics) what the units compute; this section is the fine print —
a datapath body is a chain of unit `with`-blocks, and **each block accepts only a narrow,
fixed shape of code**. Most authoring errors are contract violations, so learn them up front.

This section goes: the **contract table** (every block at a glance) → **how the blocks
interact** (topology + IFD feeder routing) → the two movement blocks whose contracts
are most often violated, **IFD** and **ODR** → and **DPU register programs** ([§8.5](#85-dpu-register-programs)). The other
compute block, `MatMul`, is covered by the table below plus the
skeletons and intrinsics in [§6](#6-the-datapath-model)–[§7](#7-compute-intrinsics).

### 8.1 The block contract table

| Block | Role | The contract (one line) |
|---|---|---|
| `A.IFD(num=…)` | input feed | affine address gen + optional PWORD pad; a **single `stream.push`**; no compute |
| `A.pseudo.MatMul` | multiply | reset the accumulator, `acc += A.mvm_1x1(stream.pop(), w_imc[kb, nb])` over a reduction of **compile-time-constant length**, push acc |
| `A.DPU` | elementwise | preload registers (init), then a program of register ops (cast/lut/fma/reduce/arith), push |
| `A.ODR` | output drain | pop → optional `cast`/`vtrsp` → store at affine address, **or drop** on a predicate; no compute |
| `A.CoreDMA` | copy via the AI core's own DMA unit | affine buffer→buffer copy (e.g. `out[i]=in[i]`), no compute/cast; can move data on- or off-core |
| `A.SysDMA` | copy via the system DMA unit | affine buffer→buffer copy, no compute/cast |

`A.pseudo.LoadIMC` is not a block inside a datapath but a datapath of its own, written outside any
`A.Datapath`. Its contract: one store per IMC block, of a 64-row slice of the weights —
`w_imc[kb, nb] = weights[kb * 64 : kb * 64 + 64, nb]` ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)). Both it and the MatMul index the IMC
buffer with **two** indices, `w_imc[k_block, n_block]`; write that form, as in the examples above.

### 8.2 How the blocks interact — topology and IFD feeder routing

Within a datapath the blocks form a **fixed pipeline**, and the `num` in `A.IFD(num=…)` selects
the feeder. It is **not interchangeable** — it indexes that datapath kind's feeder set, each
wired to a specific downstream unit — and it is **optional**, defaulting to `0` (the main
feeder); pass `num=1` only for the DPU side-input. The canonical per-datapath unit sequence is:

```text
MVMEXE datapath:
  ┌────────┐   ┌────────┐   ┌─────┐   ┌─────┐
  │ IFD(0) │──▶│ MatMul │──▶│ DPU │──▶│ ODR │
  └────────┘   └────────┘   └──▲──┘   └─────┘
                               │
                  ┌────────┐   │
                  │ IFD(1) │───┘   (optional DPU side-input)
                  └────────┘

DWPU datapath:   same shape, with the depthwise stage in place of MatMul (§6.3).
```

- **`IFD(num=0)`** is the main input, and the **only** feeder that can reach the main compute
  stage — **MatMul** (MVMEXE datapath) or the depthwise stage (DWPU datapath). It can also feed
  the DPU directly when that stage is bypassed ([§6.4](#64-bypass--writing-only-the-units-you-need)), which is what a DPU-only elementwise op
  does.
- **`IFD(num=1)`** is the **side-input feeder**, and the DPU is its only consumer: a matrix or
  depthwise stage may not pop it. Use it for a second operand — an FMA, a bias, a second tensor.
  Optional; present only when the DPU needs a side input.

`IFD(num=0)` and `IFD(num=1)` are read the same way — `stream.pop()`, in the block that consumes
them. The difference is *which unit may consume each*, not how you pop it.

**What `num` may be.** `num` is an index into that datapath kind's feeder set, so its legal
values follow from the diagram: `0` or `1`, in either kind. An out-of-range value is rejected at
lowering. A `LoadIMC` datapath has no feeder for you to number — its weights are fed for you
([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)).

### 8.3 IFD — input feed

#### The single-StreamPush rule

The body of a `with A.IFD():` block must reduce to **exactly one `stream.push(...)`** whose
value is one of: a plain `stream.push(buffer[idx])`, the bounds-checked padded form below,
or an all-pad `stream.push(A.Broadcast(pad, 64))`.

```python
with A.IFD(num=0):
    ifd_stream.push(
        A.if_then_else(cond,                             # in-bounds predicate
                       A.BufferLoad(input_buffer, idx),  # real load
                       A.Broadcast(A.int8(0), 64))       # neutral pad when out of bounds
    )
```

**What `cond` looks like.** One `idx >= 0` / `idx < extent` pair per guarded axis, combined with
`tir.all`, where `idx` is the input position the output position maps to:
`idx = out_pos * stride + tap * dilation - pad`. A padded 2-D convolution guards both spatial axes:

```python
def _padded_load(oh, ow, kh, kw, cib):          # plain Python helper → returns an expression
    ih = oh * STRIDE_H + kh * DILATION_H - PAD_H
    iw = ow * STRIDE_W + kw * DILATION_W - PAD_W
    cond = tir.all(ih >= 0, ih < H, iw >= 0, iw < W)
    return A.if_then_else(cond,
                          A.BufferLoad(inp, (ih, iw, cib)),
                          A.Broadcast(A.int8(0), 64))     # 0 for conv, -inf for maxpool

with A.IFD(num=0):
    ifd_stream.push(_padded_load(oh, ow, kh, kw, cib))
```

The helper is what keeps the block a single `stream.push` — see the caution below. This is the
shape the operator library uses for every padded op.

> **Caution — "Expected a single StreamPush".** Do **not** introduce intermediate
> assignments inside the IFD block — each becomes a `LetStmt` (an intermediate binding), which is
> not allowed in an IFD block. Build any index/condition math in a **Python helper that returns the
> expression**, then pass it straight to `.push(...)`.

Rules: the address must be **linear** in the loop vars; the value reaches the stream exactly as
loaded — an IFD block casts nothing, so give the load the dtype the consumer wants; the pad
`Broadcast` must have **64 lanes**; no compute on the loaded data. The compiler lowers the
`if_then_else(cond, load, Broadcast)` shape into offset/width bounds on the load,
using the pad value **you supply in the `Broadcast`** (e.g. `0` for conv, `-inf` for maxpool).

#### Several streams on one feeder are read in order, not interleaved

`A.decl_stream(dtype, scope)` lets you declare **any number** of logical streams and feed
them from IFD blocks, so the program *looks like* the consumer has N independent inbound
channels. It does not: a datapath's feeder set is fixed
([§8.2](#82-how-the-blocks-interact--topology-and-ifd-feeder-routing)), and when one feeder
carries several logical streams it delivers them **strictly in order — all of the first, then all
of the second, never interleaved**.

> **Caution — undefined.** If two logical streams share one feeder and the consumer pops them in
> an interleaved order (`a0, b0, a1, b1, …`), the read pattern is **undefined**: it is not
> compiler-checked, and the program either stalls waiting for an order that never arrives, or
> returns **incorrect values** (elements read in the wrong order). Multiple *sequential*,
> possibly differently-typed segments on one feeder are legitimate — only *interleaved* reads
> across streams that share a feeder are **undefined**.

```python
# The following is WRONG and leads to undefined behavior:
# ✗ looks like two independent inbound channels — but both bind to IFD_1 (one feeder)
sa = A.decl_stream("int8x64", A.IFD_1_STREAM)
sb = A.decl_stream("int8x64", A.IFD_1_STREAM)
with A.IFD(num=1):
    for i in range(4): sa.push(buf_a[i])      # feed a
with A.IFD(num=1):
    for i in range(4): sb.push(buf_b[i])      # feed b
with A.DPU():
    for i in range(4):
        x = sa.pop(); y = sb.pop()            # author assumes the pair (a_i, b_i)
# one feeder drains ALL of sa, then ALL of sb → wire order: a0 a1 a2 a3 b0 b1 b2 b3
# so the pops actually pair (a0,a1) (a2,a3) (b0,b1) (b2,b3)  → wrong data (or a stall)

# ✓ workaround: interleave upstream (in memory), then drain a single stream straight through
s = A.decl_stream("int8x64", A.IFD_1_STREAM)  # buf_ab = [a0,b0,a1,b1,…] laid out in memory
with A.IFD(num=1):
    for i in range(8): s.push(buf_ab[i])
```

### 8.4 ODR — output store / drop

Unlike IFD (which pads with the `A.if_then_else` value-select, [§8.3](#83-ifd--input-feed)), ODR chooses **store vs
drop** — a statement-level choice, not a value — so it cannot use `A.if_then_else`; it uses
`A.If`/`A.Then`/`A.Else` with `A.drop` instead. One predicate guards it, on the `A.If`:

```python
with A.ODR():
    with A.If(pred):                 # selects store or drop
        with A.Then():
            output[out_idx] = dpu_stream.pop()
        with A.Else():
            A.drop(dpu_stream.pop())  # surplus positions are dropped
```

> **Store by indexing the operand**, as above — not with `A.buffer_store(output.buffer, …)`.
> `A.buffer_store` takes a `Buffer`, so reaching an operand's `.buffer` is the only way to pass one,
> and that discards the operand's position: if the caller supplied a sub-region, the store lands at
> the parent's origin instead ([§4.3](#43-operators-as-the-usual-authoring-path-scriptop--arg), [§14](#14-correctness-hazards--limits)). Indexed assignment goes through the operand and resolves
> correctly. Keep `A.buffer_store` for cases where you genuinely hold a `Buffer` — a DPU register
> file, as in [§8.5](#85-dpu-register-programs).

> **Caution — the pop appears in both branches.** The stream pop is one action
> routed to store-or-drop. The store value may be wrapped in `A.cast` and/or `A.vtrsp`
> (vector transpose; `vtrsp` only on packed dtypes `int8x64`/`float16x32`/`float32x16`).
> **`A.vtrsp` is only available on the DWPU datapath (`A.DP_DWPU`), not on the MVMEXE
> datapath (`A.DP_MVMEXE`).**

A whole transpose is written through the operator library: `axol.core.transpose(x, dims=...)`
permutes any axes, and `axol.atomic.transpose` / `axol.atomic.vector_transpose` do it in a single
datapath.

A cast is legal here, but the DPU is where conversions normally go; prefer it unless you have a
reason not to.

`A.drop(...)` is the primitive **restricted to the ODR (drop) pattern** — it is how you
discard surplus stream positions. `CoreDMA` may carry an `A.where(cond)` region predicate.

> **You may see `A.where` on an ODR block in existing operators; it has no effect there.** Block
> predicate takes effect only on the top-level blocks — `A.Datapath`, `A.CoreDMA`, `A.SysDMA`
> ([§10](#10-predicates-and-conditionals)) — and `ODR` is a stream unit, so the predicate is never applied; the compiler reads
> the `A.If` instead. It
> is redundant, not wrong; write the `A.If` alone.

### 8.5 DPU register programs

A `with A.DPU():` block is a short program over **registers** — `A.var(dtype)` for a single
register (read/written as `x`), `A.array(dtype, N)` for an `N`-element file (read/written as
`regs[i]`); scope is inferred from the dtype ([§5.2](#52-allocation)). The idioms:

- **write** by assignment — `x = expr` or `regs[i] = expr`; shipped operators instead write the
  explicit `A.buffer_store(regs, expr, (i,))` and read `regs[i]`, which is the same thing;
- **read** by name/index — `x`, `regs[i]` — and combine reads with the operators `+ - *` and the
  DPU ops `A.dpu.ops.*` (`lut`, `prelu`, `fma`, `add_reduce` / `max_reduce`, …);
- **`with A.init():`** is a one-time preamble that runs **once before the streaming
  iterations** — use it to preload constants that later iterations only read; a register it
  writes cannot be written again in the loop ([§14](#14-correctness-hazards--limits));
- **push** results to the output stream.

The example below applies a per-channel affine transform: `A.init` loads a `scale` and a `bias`
once (constants arrive on the IFD1 side-input feeder, [§8.2](#82-how-the-blocks-interact--topology-and-ifd-feeder-routing)), and each streamed input is scaled
and shifted by them. Registers that a preamble fills and later iterations read must be `A.array`,
not `A.var` — `A.var` is within-iteration scratch.

```python
with A.DPU():
    scale = A.array("float18x64", size=1)      # loaded once in the preamble, read every iteration
    bias  = A.array("float18x64", size=1)
    acc   = A.array("float18x64", size=1)      # written and read within one iteration
    with A.init():                             # preamble — runs once, before the per-element loop
        scale[0] = A.cast(ifd1_stream.pop(), "float18x64")   # constants on the side-input feeder
        bias[0]  = A.cast(ifd1_stream.pop(), "float18x64")
    x = A.var("float18x64")                               # declared: it is read by a later statement
    x = A.cast(ifd0_stream.pop(), "float18x64")           # each iteration: read the main input,
    acc[0] = x * scale[0]                                 #   scale …
    dpu_stream.push(A.cast(acc[0] + bias[0], out.dtype))  #   … then shift — one op per statement
```

> **One arithmetic operation per statement.** A DPU statement carries at most one `+`, `-` or `*`.
> Writing the affine transform as a single `x * scale[0] + bias[0]` is **refused**, and the error
> names the enclosing statement rather than the expression (`Failed to parse TIR statement …`), so
> split it as above. Both operands of that one operation must also already be `float18x64` ([§7](#7-compute-intrinsics)).

**A reduction inside one DPU block.** An accumulator the block updates is seeded in the body, from
the first value, not in `A.init`. Read the whole reduction into a register file, then combine it —
the shape a row sum, a softmax or a norm takes:

```python
with A.DPU():
    x   = A.array("float18x64", size=NB)          # the row, one register per block
    acc = A.var("float18x64")
    for j in A.serial(NB):
        x[j] = A.cast(ifd_stream.pop(), "float18x64")
    acc = x[0]                                    # seed from the first value
    for j in A.serial(1, NB):
        acc = acc + x[j]                          # combine the blocks elementwise
    dpu_stream.push(A.cast(A.dpu.ops.add_reduce(acc), "float32x64"))   # then across the lanes
```

A reduction longer than the registers hold is staged through memory, across datapaths
([§14](#14-correctness-hazards--limits)).

---

## 9. Loops: build-time unrolling and loop lowering

With the units and their contracts settled, the remaining question is iteration. Loops in
AxeleraScript mean different things depending on **where** they sit and **what they wrap**. A plain-Python `for` outside the parsed script unrolls at build time ([§9.1](#91-the-two-for-rule)). A device
loop written *around* a datapath is rewritten by the compiler — how, depends on how many
datapaths it wraps, and this split is a **deliberate trade-off**:

- **one datapath → the loop is *sunk into* the units** ([§9.2](#92-one-datapath-in-the-loop--the-loop-is-sunk-into-the-datapath)), becoming that datapath's
  streaming loop. Wrapping a single datapath is also what makes the loop a **tiling candidate** —
  the compiler reads its extent and can split it into tiles with DMAs inserted ([§11](#11-tiling)).
  Tiling reads the loop before sinking rewrites it, so the two do not interact ([§11.1](#111-what-can-be-tiled)).
- **two or more datapaths → the loop is *unrolled*** ([§9.3](#93-two-or-more-datapaths-in-the-loop--the-loop-is-unrolled)): there is no single pipeline to sink
  into, so the compiler keeps the loop and replicates its body per iteration — giving up the
  tileable single-pipeline form in order to sequence work across the datapaths.

### 9.1 The two-`for` rule

Two `for` loops that look identical behave differently depending on **where** they sit:

```python
for c in range(cores):            # OUTSIDE the parsed script → runs at parse time, unrolls to `cores` copies
    with A.script():
        for i in A.serial(chunk):  # INSIDE the parsed script → ONE device loop
            with A.CoreDMA(core=c):
                out[c * chunk + i] = inp[c * chunk + i]
```

A `for … in range(...)` in **plain Python** (outside a parsed region) runs at parse time and
unrolls the block into copies — this is metaprogramming. A `for` **inside** the parsed script is
a device-side construct: `A.grid` / `A.serial` / `A.unroll` (the last is itself fully unrolled
at lowering, not a surviving loop — [§4.4](#44-device-loop-constructs)), and a bare `range` too (inside the script,
`for i in range(n)` is the same as `for i in A.serial(n)`). Confusing the two is a common
source of bugs.

### 9.2 One datapath in the loop → the loop is *sunk into* the datapath

```python
# BEFORE — the author writes one loop around a single datapath
for oh, ow in A.grid(OH, OW):
    with A.Datapath(kind=A.DP_MVMEXE):
        with A.IFD(num=0):      ifd_stream.push(inp[oh, ow])
        with A.pseudo.MatMul(): ...
        with A.ODR():           out[oh, ow] = dpu_stream.pop()

# AFTER — the loop is sunk into each unit; the datapath appears once
with A.Datapath(kind=A.DP_MVMEXE):
    with A.IFD(num=0):      for oh, ow in A.grid(OH, OW): ifd_stream.push(inp[oh, ow])
    with A.pseudo.MatMul(): for oh, ow in A.grid(OH, OW): ...
    with A.ODR():           for oh, ow in A.grid(OH, OW): out[oh, ow] = dpu_stream.pop()
```

A unit that already has its own inner loop — e.g. the IFD streaming a reduction — keeps that
loop nested inside the sunk one, so its loop nest is deeper than the others'.

That "all pushes, then all pops" shape is safe even though it puts far more than a stream's worth
of elements in flight: a full stream stalls its producer
([§14](#14-correctness-hazards--limits)). You do not need to bound the in-flight count to
make sinking correct.

**A unit appearing twice keeps both blocks, in order.** Sinking rewrites **each unit block
independently**, giving every one its own copy of the loop, and it does not group or reorder blocks by
unit. So a datapath written as `IFD → DPU → IFD → DPU → ODR` stays in that order after sinking,
with the loop replicated into each block. Nothing hoists all the blocks of one unit ahead of another's.

> **Where that replication becomes a correctness hazard.** Each block gets its *own* copy of the loop,
> so the emitted order is *block 1 for every iteration, then block 2 for every iteration* — never
> *(block 1, block 2) per iteration*. A unit that **carries state from one of its blocks to the
> next** therefore sees only the last iteration's state, with no error reported. The fix is
> structural: split the stateful phases into **separate datapaths**, so the loop wraps two
> datapaths, which [§9.3](#93-two-or-more-datapaths-in-the-loop--the-loop-is-unrolled) unrolls rather than sinks — or loop at build time ([§9.1](#91-the-two-for-rule)).

A loop is sunk only if it wraps **exactly one** datapath **and is of serial kind**; sinking
proceeds innermost→outermost and stops at the first shared or non-serial loop. Both `A.serial`
**and `A.grid` qualify** — an `A.grid` nest is a nest of serial loops, so each level is a candidate.
`A.unroll` is not serial, which is exactly what makes it the opt-out described below.

```python
# Sinking goes innermost→outermost and STOPS at the first non-serial or shared loop:
for u in A.unroll(2):                  # A.unroll → not sunk (it is unrolled); sinking stops here
    for i in A.serial(N):              # serial, wraps ONE datapath → sunk into the units
        with A.Datapath(kind=A.DP_MVMEXE):
            ...
```

So `A.unroll` is the explicit **opt-out**: wrap a single datapath in `A.unroll` (instead of
`A.serial` / `A.grid`) and the loop is unrolled in place rather than sunk. Reach for it when you
want the iterations expanded in place instead of streamed. Note it is not a remedy for the hazard
above: split the datapath instead.

### 9.3 Two or more datapaths in the loop → the loop is *unrolled*

```python
for i in range(N):
    with A.Datapath(): ...   # ONE datapath  → loop sunk INTO the units (disappears)

for i in range(N):
    with A.Datapath(): ...   # TWO datapaths → SAME loop stays, then is UNROLLED
    with A.Datapath(): ...
```

A loop containing ≥2 datapaths is a **shared loop**: it genuinely sequences work across
datapaths and cannot be sunk into a single unit. Such a loop is instead **fully unrolled**.

> **Easy to misread:** identical syntax, opposite outcome, decided by a non-local property
> (one datapath vs many in the loop body). You cannot tell from the loop itself what it
> will do. If you want a datapath repeated, understand it will be *distributed into the
> units*, not re-run as a block.

---

## 10. Predicates and conditionals

AxeleraScript's conditionals sit at different levels — pick by what you need to vary: an
**expression** that selects a value, **statement** blocks that pick a code path, a **region
guard**, or a plain Python `if` resolved at **build time**.

- `A.if_then_else(cond, a, b)` — a value-level **select**: an *expression* that yields `a`
  where `cond` holds and `b` otherwise (`a` and `b` are same-dtype values, not statements).
  Use it to choose a value; use `A.If` to choose a code path. It is the IFD pad pattern ([§8.3](#83-ifd--input-feed)).
- `with A.If(c): with A.Then(): ... with A.Else(): ...` — a statement-level conditional block
  (chooses which body executes at runtime, not build time), as opposed to `A.if_then_else`
  which selects a value; its primary use is the ODR store-or-drop pattern ([§8.4](#84-odr--output-store--drop)).
- `with A.when(cond):` — shorthand for `with A.If(cond): with A.Then(): ...` (no else); gates
  a region on a runtime predicate (e.g. software-pipeline stages).
- `A.where(cond)` — records a **block predicate**: a region guard on the enclosing block. It
  bounds generated data movement, and takes effect on the **top-level** blocks — `A.Datapath`,
  `A.CoreDMA`, `A.SysDMA` — only. On a stream-unit block such as `A.ODR` it is inert ([§8.4](#84-odr--output-store--drop)).
- Python-level `if` on **compile-time-known** values selects a branch at *build time*
  (e.g. `if self.has_bias:` picks a datapath variant) — it never becomes a runtime branch.
  Inside a parsed region it behaves differently in one way worth knowing before you rely on it:
  both arms are parsed ([§4.2](#42-the-parsed-form-and-what-it-accepts)).

**Two surfaces for the same idea.** A region predicate appears either as a **statement** at the top
of a block — `A.where(cond)` — or as a
**keyword on an operator call**, `where=cond`, used to gate a
software-pipeline stage ([§12](#12-synchronization-and-fusion)). Passing `where=` to a call is equivalent to wrapping that call in
`with A.when(cond):`; pick whichever reads better at the call site — except inside a **pipelined**
tile loop (`A.tiles(..., drain=N)`), where a stage must be gated by the `where=` predicate: the
scheduling machinery that staggers the stages does not carry a conditional statement through. A
sync is not a call and has no `where=`, so it stays wrapped in `A.when` ([§12](#12-synchronization-and-fusion)).

**Prefer predicates over duplicating kernel bodies** for shape/flag specialization.

---

## 11. Tiling

**Tiling** splits a kernel's work into tiles that each fit the small memories (IMC /
L1 / L2): DMA a tile in → run the datapath → DMA the result out, then repeat for the next tile.
It is **advisory and automatic by default** — you write ordinary loops and the compiler tiles
them for you; the `A.tiles` API below is only for *directing* it.

### 11.1 What can be tiled

Tiling acts on a kernel's **outer loops**, the ones *outside*
a datapath: such a loop is a **tile loop**, and the compiler can split it so each iteration
processes one tile. A loop *inside* `with A.Datapath()` is the units' **streaming
loop** ([§9.2](#92-one-datapath-in-the-loop--the-loop-is-sunk-into-the-datapath)) and is never tiled — whether you wrote it inside a unit block or around one, as
[§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)'s reduction loop wraps its `A.IFD` block. So keeping a dimension as an outer loop, outside
the datapath, is what makes it tileable.

Tiling happens **before** outer loops are moved into the units ([§9.2](#92-one-datapath-in-the-loop--the-loop-is-sunk-into-the-datapath)), so it is the code you wrote
that decides what can be tiled, not the shape the datapath ends up with. A loop that [§9.2](#92-one-datapath-in-the-loop--the-loop-is-sunk-into-the-datapath) sinks was
already a tiling candidate; sinking neither creates nor removes the opportunity.

Two structural cases leave nothing to tile, and neither reports anything:
- a dimension **unrolled at build time** (a plain-Python `for`, [§9.1](#91-the-two-for-rule)) has no loop left to split —
  the tileable dimensions are the device loops you kept;
- a **device loop enclosing two or more datapaths** is unrolled rather than split ([§9.3](#93-two-or-more-datapaths-in-the-loop--the-loop-is-unrolled)), so that
  loop is not a candidate.

Tiling is advisory, so in both cases you get untiled code.

**Blocks that share a buffer tile together.** The compiler joins the tiling of blocks that read and
write the same buffer along the same dimension, so a load, a compute and a store over one tile
advance as a single pass — the shape the `A.tiles` examples in [§11.2](#112-directing-it--atiles) and [§12](#12-synchronization-and-fusion) are built on.

**An execution limit can force tiling even when memory would not.** A compute unit's own capacity can
cap how much of a dimension one datapath or one residency may cover, independently of how much fits
in L1 or L2 — most sharply the IMC tile budget ([§14](#14-correctness-hazards--limits)), which caps how much of a matrix one
residency holds. An operator over a larger matrix must therefore split it along K whether or not
memory is tight, and the operator is responsible for that split.

> **`A.tiles` splits a dimension into *disjoint* pieces — no argument makes them overlap.** A sliding
> window needs consecutive input pieces to overlap by `filter - stride` elements, so tile the
> **output** and derive the input region from the tile handle: base `tile.base · stride`, extent
> `(tile.lanes - 1) · stride + filter`. The overlap is expressed by that derivation, not by the
> split — which is how the shipped windowed operators tile, and the operator library wraps it.

By default you write those outer loops as ordinary device loops (`A.grid` / `A.serial`) and let
the compiler split them — analyzing the access pattern and inserting the DMAs. Most operators
rely entirely on this and write no tiling of their own:

```python
for i in A.grid(M):                         # OUTSIDE the datapath → tiling candidate; the compiler
                                            #   may split this into tiles (and insert the DMAs)
    with A.Datapath(kind=A.DP_MVMEXE):
        with A.IFD(num=0):
            for k in A.serial(K):           # INSIDE the datapath → not tiled (streaming loop)
                ifd_stream.push(...)
        with A.pseudo.MatMul(): ...         # consume the stream and compute
        with A.ODR(): ...
```

### 11.2 Directing it — `A.tiles(...)`

When you want to choose which dimension to tile,
bound the tile factor, or request software pipelining, wrap the outer loop in the `A.tiles(...)`
context manager — the compiler still picks the concrete factor within your bounds. It takes the
dimension's `extent`, yields a **tile handle** (a ramp; read `.lanes`), and you slice buffers
with `buf.tile(handle, dim=N)` (chainable):

```python
with A.tiles(m) as m_tile:                          # `m` = extent of the dimension to tile
    atomic.full((m_tile.lanes, n_blocks), 0.0, output=c_l1.tile(m_tile, dim=0))  # zero the K sum
    atomic.copy(a_2d.tile(m_tile, dim=0), output=a_l1.tile(m_tile, dim=0))   # DMA the input tile in
    with A.tiles(k_blocks, tile_min=k_min, tile_max=k_max) as k_tile:   # nest to tile more dims
        k_rows = A.Ramp(k_tile.base * 64, 1, k_tile.lanes * 64)          # this K tile's weight rows
        atomic.copy(b_2d.tile(k_rows, dim=0), output=b_l1.tile(k_rows, dim=0))
        with A.tiles(n_blocks) as n_tile:
            atomic.load_imc(b_l1.tile(k_rows, dim=0).tile(n_tile, dim=1),
                            imc.tile(k_tile, dim=0).tile(n_tile, dim=1))  # make this tile resident
            atomic.matmul(matrix_ifd=a_l1.tile(m_tile, dim=0).tile(k_tile, dim=1),
                          matrix_imc=imc.tile(k_tile, dim=0).tile(n_tile, dim=1),
                          partial=c_l1.tile(m_tile, dim=0).tile(n_tile, dim=1),  # add onto the sum
                          output=c_l1.tile(m_tile, dim=0).tile(n_tile, dim=1))
    atomic.copy(c_l1.tile(m_tile, dim=0), output=c_2d.tile(m_tile, dim=0))   # DMA the output tile out
```

**Splitting K accumulates through `partial=`.** Each `atomic.matmul` adds its product onto the
`partial` buffer and writes the sum to `output`; pass the same region for both, and zero it once
before the K loop, as above. Without `partial=`, each K tile overwrites the one before it — silently.
Each K tile is also loaded into the IMC before the matmul that reads it (`atomic.load_imc`).

`extent` is the build-time dimension size; the loop is symbolic and the compiler substitutes
the concrete tile factor. You then either **bound** that choice or **make** it:
- `tile_min` / `tile_max` — bound the factor, leaving the compiler to pick within your range;
- `num_tiles=N` — fix the split outright at `N` tiles, leaving nothing to choose.

`drain=N` requests software pipelining (runs `N` extra steps so a tile's compute overlaps the next
tile's DMA load; see [§12](#12-synchronization-and-fusion), which also covers the buffering that requires).

**The last tile is short, and the handle says so.** The handle is a ramp with `base = i · tile_size`
and `lanes = min(tile_size, extent - base)`, so when the extent does not divide evenly the final
tile's `.lanes` is the remainder — an uneven split needs no predicate of yours. Read `.lanes` for the
real size, and remember it is *not* the nominal tile size on that last iteration.

---

## 12. Synchronization and fusion

Synchronization and fusion are both **advisory** — you state intent, the compiler decides — and
both are applied **automatically**. Hand-written sync or fusion is the exception, not the norm.

### Synchronization

By default every launch — a datapath, a DMA, an operator — is followed by a wait for it, so work
runs in program order, and within a datapath a unit stalls until its stream input arrives
([§6](#6-the-datapath-model)). You hand-write sync to overlap work that program order would serialize, or to order
work across cores — chiefly **software pipelining**, **running the two datapath kinds in
parallel**, and **cross-core** coordination. A hand-written wait replaces the compiler's for the
launch it names, so ordering that launch against the work after it is then up to you.

Sync works on **ids**: tag a unit of work (an op or a datapath) with `id="..."`, then order the
tagged work.

```python
A.sync(waits=[...], signals=[...], buffers=[...])  # wait on / signal ids; buffers = the data guarded
A.waits(*ids)                                      # shorthand for sync(waits=ids)
A.signals(*ids)                                    # shorthand for sync(signals=ids)
A.barrier(*cores)                                  # all-to-all sync across ≥2 cores (no-op for <2)
```

- `waits` = ids that must finish first; `signals` = ids this point marks done; `buffers` = the
  buffers the sync guards, so the compiler ties it to the right data.
- **Simplest use** — the wait the compiler would insert anyway, written out; writing it yourself
  is how you move it later, past other work:
  ```python
  copy(a_ddr, output=a_l1, id="load_a")   # tag the load
  A.sync(waits=["load_a"])                 # the matmul below waits for it
  ```
- **Running the two datapath kinds in parallel** ([§2](#2-the-execution-model-that-the-language-targets)) —
  tag each datapath and wait for them only after both are launched, one wait per id:
  ```python
  with A.Datapath(id="mm"): ...           # a matmul → DP_MVMEXE
  with A.Datapath(id="ew"): ...           # independent elementwise work → DP_DWPU
  A.waits("mm")                           # both run while the core waits here
  A.waits("ew")
  ```
  The two must be independent. Placing the waits after both launches also removes the ordering the
  compiler would otherwise insert, so a datapath that reads what the other writes can run before
  the write lands — wait on the producer before launching its consumer.

**Software pipelining** is the main reason to hand-write sync — overlap a tile's compute with the
next tile's load. `A.tiles(..., drain=N)` runs `num_tiles + N` steps so the stages stagger;
a `where=` predicate gates each stage on the current step; each stage carries an `id`; `A.waits`
orders them, wrapped in `A.when` since a sync takes no `where=` ([§10](#10-predicates-and-conditionals)):

```python
with A.tiles(n_tiles, drain=2) as t:          # +2 drain steps → load/compute/store overlap
    copy(...,   id="load",    where=is_load)            # DMA the next tile in
    matmul(..., id="compute", where=is_compute)         # compute on the current tile
    copy(...,   id="store",   where=is_store)           # DMA the finished tile out
    with A.when(is_compute): A.waits("load")            # a sync takes no where= — gate it (§10)
```
(The `is_load` / `is_compute` / `is_store` predicates are derived from the pipeline step.)

> **A pipelined buffer needs one slot per in-flight step.** Because `drain=N` makes the stages
> overlap, a buffer written by one stage and read by a later one is live in several steps at once:
> the load for step *k+1* would otherwise overwrite what the compute for step *k* is still reading.
> Give it a leading slot dimension — `A.alloc_buffer((2, …))` for the two-deep pipeline above — and
> index it by step. This is a **correctness** requirement of `drain=`, not a tuning choice.

### Fusion

Fusion chains one datapath's ODR directly into the next's IFD by matching their stream/buffer
access patterns — streaming dataflow is what makes it possible. It runs **automatically** where
safe, so the blocks below are **rarely needed** — use them only to force or block a specific
merge.

```python
with A.fuse(unsafe=False, strict=True,
            disable_dpu_reg_pressure_heuristic=False, enable_loop_legalization=False):
    ...   # force the enclosed datapaths to merge into one
with A.prevent_fuse():
    ...   # block fusion inside
```

- `strict=True` — raise if the group does **not** reduce to a single datapath.
- `disable_dpu_reg_pressure_heuristic=True` — fuse even past the safety check that blocks merges
  which would overflow the DPU's registers.
- `enable_loop_legalization=True` — allow loop legalization during fusion (can interfere with
  tiling).
- `unsafe=True` — shorthand that sets both of the above.

---

## Part III — Model Authoring & Reference

This part is what you reach for once the language itself is familiar: composing existing operators
into a whole model ([§13](#13-high-level-model-authoring)), and the reference behind the rest of the guide ([§14](#14-correctness-hazards--limits)) — the ways a program
can be silently wrong, the errors that name a line other than the one responsible, and applicable
limits that stay invisible until you exceed them.

## 13. High-level model authoring

The same `@A.prim_func` syntax composes existing AXOL operators into a whole model, with
ordinary Python SSA assignments. The AXOL `aten`/`core` layer aims to follow the PyTorch
surface as closely as possible (though this is a best-effort resemblance, not a guaranteed
one-to-one mapping), so the result reads much like a `forward()`:

```python
from axelera.compiler.alto import axol       # the operator library (§4.3)
import axelera.compiler.alto.script as A

IMG_C, IMG_H, IMG_W = 64, 224, 224   # 3 image channels zero-padded to one PWORD (§5)
IMG_W_BLOCKS  = 4                    # W padded 224 → 256, packed while C is not yet innermost
PATCH_H, PATCH_W = 16, 16
HIDDEN_BLOCKS = 6                    # hidden dim 384 = 6 PWORDs
N_PATCHES     = (IMG_H // PATCH_H) * (IMG_W // PATCH_W)   # 14 * 14 = 196

@A.ir_module
class ViTPatchEmbed:                        # first 5 ops of ViT-S/16 (patch-embed stem)
    @A.prim_func
    def main(
        arg0:   A.Buffer((1, IMG_C, IMG_H, IMG_W_BLOCKS), "int8x64", scope=A.DDR),          # image (NCHW; W packed on the way in)
        w_conv: A.Buffer((HIDDEN_BLOCKS, PATCH_H, PATCH_W, IMG_C), "int8x64", scope=A.DDR), # patch-embed conv weight
        w_mean: A.Buffer((1, 1, HIDDEN_BLOCKS), "float32x64", scope=A.DDR),                 # per-channel mean
    ):
        b0 = axol.aten.permute(arg0, dims=(0, 2, 3, 1))                   # NCHW → NHWC; C becomes the packed axis
        b1 = axol.aten.slice_tensor(b0, dim=2, end=IMG_W)                 # trim padding column (256 → 224)
        b2 = axol.aten.convolution(b1, w_conv, stride=(PATCH_H, PATCH_W), # 16×16 stride-16 patch embedding
                                   padding=(0, 0), dilation=(1, 1))
        b3 = axol.aten.reshape(b2, shape=(1, N_PATCHES, HIDDEN_BLOCKS))   # (1,14,14,6) → (1,196,6)
        b4 = axol.aten.sub_tensor(b3, w_mean)                            # per-channel normalization
        A.output(b4)
```

It is **PyTorch-like but explicit and static**:
- Buffers carry **memory scopes** (`scope=A.DDR`) and **vectorized, PWORD-packed dtypes**
  (`"float32x64"`, dims `// 64`).
- **Which axis is packed follows the data**, and `permute` moves it: the image arrives packed
  along W, and the permute to NHWC makes channels innermost ([§5](#5-data-types-and-buffers)) —
  which is why the 3 image channels are padded to a full PWORD first.
- **Quantization is spelled out**, not hidden: e.g. `mul(qs) → add(qz) → dpu_cast(int8)
  → matmul → sub(dz) → mul(ds)`.
- It is a **static functional graph** — no `nn.Module`, no autograd, no dynamic shapes
  (every shape — the image size, patch count, hidden dim — is resolved at build time in plain
  Python).

---

## 14. Correctness hazards & limits

This section is for when something is wrong, and for the limits you build against. It holds four
things: an index from the error you are holding to its cause, the failure modes that report
nothing at all, the invariants your program has to uphold, and the limits the compiler enforces
for the target you build for.

### Index by symptom — start here if you have an error

Find the error you are holding, then read its entry below. The error strings are quoted as
emitted, so they can be searched for verbatim. **The middle column is where the error
*points*, which is usually not where the problem is** — that mismatch is what makes these
expensive.

| What you see | Where it points | Actual cause |
|---|---|---|
| `Failed to parse TIR statement` | the compute block | a Python name read by a later statement; values crossing a statement boundary must be registers (`A.var` / `A.array`, [§5.2](#52-allocation)) |
| `Failed to parse TIR statement` | the compute block | more than one arithmetic operation in a single DPU statement; split it, one op per statement ([§8.5](#85-dpu-register-programs)) |
| `Cannot index acc with [0]` / `Cannot store into acc at 0` | the indexed read / the indexed write | the MVM accumulator and `A.var` are registers, used without an index ([§5.2](#52-allocation), [§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)) |
| `The MVM accumulator only supports 'acc += A.mvm_1x1(...)'` | the accumulate itself | an accumulate that is not `acc += A.mvm_1x1(...)`, e.g. `acc *= …`; the other deviations from the reset / reduction / push form each get their own message naming the rule ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)) |
| `Expected a single StreamPush` | the IFD block | intermediate `LetStmt` in an IFD block; build the expression in a Python helper and push the result ([§8.3](#83-ifd--input-feed)) |
| `Expected linear index` | at lowering | non-affine addressing in a movement unit, e.g. `buf[(i * 5) % 8]` |
| `Cannot match type handle vs int32` | memory scheduling, no source line | a data-dependent index — a buffer value used as an index — in a movement unit |
| `expected Object but got str`, at `def main(` | the signature | `from __future__ import annotations` (PEP 563) at module scope, with a parameterized signature ([§4.2](#42-the-parsed-form-and-what-it-accepts)) |
| `A nested 'def' is not valid AxeleraScript` | the nested `def` | a `def` inside a parsed region; define the helper outside it and call it from inside ([§4.1](#41-the-build-time--runtime-boundary), [§4.2](#42-the-parsed-form-and-what-it-accepts)) |
| `Function must be decorated` | the `def` itself | an undecorated method in an `@A.ir_module` class — a genuine missing decorator, not nesting ([§4.2](#42-the-parsed-form-and-what-it-accepts)) |
| `stored in the loop section after being initialized in the init section` | the DPU block | a register written in `A.init` and written again in the loop, e.g. an accumulator seeded in `A.init`; seed it in the loop body instead ([§8.5](#85-dpu-register-programs)) |
| `Padding is not yet supported for` | the depthwise or pooling call | a nonzero `padding` on a core depthwise or pooling op ([§6.3](#63-depthwise-operators)) |
| `Cannot tile` / `Cannot fit kernel` | the depthwise or pooling call | a kernel, dilation and stride too large for the op to tile ([§6.3](#63-depthwise-operators)) |
| parser crash, no useful message | the `prim_func` | Python `None` bound to a `prim_func` variable — including in a never-taken `if` branch |
| stall, or wrong values, no error at all | nowhere | IFD interleave hazard: two logical streams sharing one feeder, popped interleaved ([§8.3](#83-ifd--input-feed)) |
| reads or writes the wrong region, nothing faults | nowhere | a sub-region operand's offset dropped; the address stays in bounds ([§4.3](#43-operators-as-the-usual-authoring-path-scriptop--arg)) |
| only the last iteration's state is seen, no error | nowhere | state carried between two blocks of one unit inside a sunk loop; each block runs every iteration before the next block starts ([§9.2](#92-one-datapath-in-the-loop--the-loop-is-sunk-into-the-datapath)) |
| a K-tiled matmul keeps only the last K tile, no error | nowhere | `atomic.matmul` called per K tile without `partial=`; each tile overwrites the last ([§11.2](#112-directing-it--atiles)) |
| wrong MVM results (matmul / conv / linear), no error at all | nowhere | an IMC matrix reused after a different one was used on the same core; loading the second ends the first's residency ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)) |
| stale data read, no error | nowhere | hand-written waits placed after two dependent datapaths; the consumer launches before the producer finishes ([§12](#12-synchronization-and-fusion)) |

The last six rows have no error string on purpose: a *silent* wrong answer is a known failure
mode here, not necessarily a bug in your kernel.

**Correctness hazards.** Grouped by *how* they fail — most are easy to miss because they do
not produce a clean authoring-time error:

*Silently wrong, or hangs, with no error raised:*
- **IFD interleave hazard** ([§8.3](#83-ifd--input-feed)) → **undefined**: a stall, or elements
  read in the wrong order; not compiler-checked.
- **Dropping a sub-region operand's offset** → reads or writes the wrong region, with no error:
  the address stays in bounds, so nothing faults. Happens when an op materializes a sliced operand
  using the region's `shape` against the parent's data ([§4.3](#43-operators-as-the-usual-authoring-path-scriptop--arg)).
- **State carried between two blocks of one unit, inside a sunk loop** → only the last
  iteration's state reaches the second block, with no error reported: sinking runs each block for
  every iteration before the next ([§9.2](#92-one-datapath-in-the-loop--the-loop-is-sunk-into-the-datapath)). Split the phases into datapaths of their own.
- **Reusing an IMC matrix after a different one was used** → the MVM compute reads the wrong
  weights, with no error: loading a second matrix on the same core ends the first's residency,
  and with `core` unset the compiler picks the core, so the collision is not visible in the
  source. The functional interpreter models each buffer separately and computes the right answer,
  so this appears only when the program runs. Reload the matrix immediately before each reuse.
- **A K-tiled matmul without `partial=`** → each K tile's product overwrites the one before it,
  with no error. Zero the output once before the K loop and pass it as `partial=` too ([§11.2](#112-directing-it--atiles)).
- **Waiting after two dependent datapaths** → the consumer reads stale data, with no error: a
  hand-written wait placed after both launches replaces the ordering the compiler would insert
  between them ([§12](#12-synchronization-and-fusion)). Wait on the producer before launching its consumer.

*Hard failure:*
- **A Python name read by a later statement in a compute block** → `Failed to parse TIR statement …
  T.LetStmt(...)`. Values that cross a statement boundary must be declared registers (`A.var` /
  `A.array`, [§5.2](#52-allocation)); a bare name becomes a `LetStmt`, which the DPU program parser refuses. Applies
  equally in a `prim_func` and in an operator's `implementation()`.
- **Binding Python `None` to a `prim_func` variable crashes the parser.** The AxScript parser
  also evaluates *untaken* `if`-branch bodies, so a `None` used even in a never-taken branch
  still crashes:
  ```python
  bias = self.bias                     # ✗ None when the optional `bias` input is absent
  if self.bias is not None:
      out[i] = out[i] + bias[i]        #   parsed even when None → crash
  ```
  Bind unconditionally to a real buffer, or inline the optional arg into the call kwargs:
  ```python
  zeros = A.alloc_constant([0.0] * 64, "float32x64", scope=out.scope())  # a real zero-filled buffer
  bias  = self.bias if self.bias is not None else zeros                  # ✓ never binds None
  out[i] = out[i] + bias[i]                                              #   (adds zero when bias absent)
  ```
- **`from __future__ import annotations` (PEP 563) at module scope breaks a parameterized
  signature** → `expected Object but got str`, reported at `def main(`. PEP 563 stringifies
  annotations, and a signature whose shape comes from build-time values in the enclosing scope
  needs them evaluated:
  ```python
  from __future__ import annotations                          # ✗ the line responsible
  ...
  def main(x: A.Buffer((1, seq_len, dim_pw), "float32x64")):  # ← the error points here
  ```
  A signature built only from literals and module-level names still parses under PEP 563, so
  the import can sit in a module harmlessly until someone parameterizes a kernel — and then it
  breaks every such `prim_func` in the file at once.

*Caught at lowering (the error names the symptom, not your source line):*
- **Intermediate `LetStmt` in an IFD block** → "Expected a single StreamPush"; build the
  expression in a Python helper and push the result directly ([§8.3](#83-ifd--input-feed)):
  ```python
  with A.IFD(num=0):
      x = buf[i * 64 + j]                 # ✗ intermediate assignment → LetStmt → "Expected a single StreamPush"
      ifd_stream.push(x)
  with A.IFD(num=0):
      ifd_stream.push(buf[_addr(i, j)])   # ✓ compute in a Python helper that returns the expression
  ```
- **Non-affine addressing** in **IFD / ODR / CoreDMA** → "Expected linear index"
  at lowering; a data-dependent index fails earlier, with "Cannot match type handle vs
  int32" and no source line. What CoreDMA *does* accept beyond the other two is an affine but
  **non-contiguous** access — it splits that into contiguous chunk transfers — but the index
  must still be affine:
  ```python
  ifd_stream.push(buf[(i * 5) % 8])    # ✗ non-affine index → "Expected linear index"
  ifd_stream.push(buf[i * 64 + j])     # ✓ affine in the loop vars
  ```

*Wrong mental model → wrong structure (no error):*
- **Confusing a build-time `for` with a device loop** — a `for … in range(...)` in the
  metaprogramming layer (plain Python, e.g. wrapping `with A.script():`) unrolls at build time
  into copies, whereas loops *inside* the parsed script (`A.grid`/`A.serial`/`A.unroll`) are
  device loops. Mixing up the two silently produces the wrong structure ([§9.1](#91-the-two-for-rule), [§4.4](#44-device-loop-constructs)).
- **One vs many datapaths in a loop behave oppositely** — one datapath is *sunk into* the
  units, two or more are *unrolled* ([§9](#9-loops-build-time-unrolling-and-loop-lowering)).

**Author obligations (invariants you must uphold for a valid program)**
- **PWORD alignment** — the vectorized data format already assumes it: values are 64-lane
  (`x64`) blocks and a PWORD-packed dimension carries a trailing `/64` block dim, so any
  dimension packed that way must be a multiple of 64 (PWORD). Ops that pack a dimension
  validate this and raise a clear error otherwise.
- **Affine, in-bounds addressing in movement units** — in IFD / ODR / CoreDMA, each buffer's
  index must independently be an **affine** function of the device-loop variables
  (Σ cₖ·loopvarₖ + const, with build-time-constant coefficients and offset, possibly negative).
  Source and destination are independent affine maps and need not match. Addresses must also be
  in-bounds and fit the offset/width bounds; use the IFD `if_then_else` pad ([§8.3](#83-ifd--input-feed)) for
  out-of-range positions.

**Limits the compiler enforces (some invisible until exceeded — validate/guard yourself)**
- **A MatMul reduction cannot overflow its accumulator**, whatever its length ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)). How much
  of K one residency holds is a separate limit — the IMC entry below.
- **MVM reduction extent must be a compile-time constant** at lowering (resolve
  symbolic tiling vars first). The author-facing IMC matrix is `int8x4096`, **2-index
  `(K_BLOCKS, N_BLOCKS)`** (matching the examples).
- **DPU:** `float18x64` registers; DPU program size is bounded (ties into the fusion
  register-pressure heuristic). A literal index past an `A.array` file's size is refused where you
  write it ([§5.2](#52-allocation)).
- **IMC weight store holds a bounded number of tiles per core.** That cap is also per buffer, not shared across buffers on one
  core: loading a second matrix ends the first's residency rather than adding to it. Layout within
  the store is automatic: you allocate `(K_BLOCKS, N_BLOCKS)` tiles ([§6.2](#62-canonical-mvm-datapath--a-matrix-multiply)) and never address the store directly. A **larger matrix is
  supported by tiling the K dimension**, which splits it across several residencies and accumulates
  the partial results through `partial=` ([§11.2](#112-directing-it--atiles)). Guard the product yourself: an over-long reduction without K-tiling fails
  with a clear error, an over-wide one surfaces later in lowering as an internal assertion.
- **Depthwise and pooling ops:** `padding` must be zero, the output dtype is `float32x64`, and a
  window too large to tile is rejected where you call the op ([§6.3](#63-depthwise-operators)).
- **Memory capacity:** the L1 and L2 capacities depend on the hardware generation, and the compiler
  enforces them for the one you build for (`AltoConfig(generation=...)`). They bound what can be *resident*, not
  what you may declare — see [§5.1](#51-memory-buffer-scopes--bufferscope): a larger buffer still compiles, provided a
  loop accesses it, because the compiler tiles that loop and moves the buffer through memory one tile
  at a time.
- **A stream holds a bounded number of elements**, and a full one applies **backpressure** rather
  than deadlocking: the producer stalls and the consumer drains it, and a unit chain has no cycle
  to deadlock on. So stream depth is not what bounds a unit program that buffers a whole reduction
  before emitting — **register capacity is**. Past what a unit's registers hold, stage the
  reduction instead of buffering it whole.

---

## Versioning

The `Version:` in the header is independent of the compiler version; it signals the nature of a
language change:
- **major** (`1.6 → 2.0`) — a breaking language change (existing kernels must be re-learned or rewritten);
- **minor** (`1.0 → 1.1`) — an additive or corrective change (a new construct, or clarified/fixed behavior);
- **no bump** for purely editorial edits.

The guide is at **v0.2**; every edit is versioned and logged.

---

## Changelog

- **v0.2** — The datapath model is described through abstract compute units (`A.pseudo.MatMul`,
  `A.pseudo.LoadIMC`) and library operators (depthwise, pooling, transpose), with the two datapath
  kinds and how they run in parallel. Clarified synchronization, K-tiling, `A.init` and DPU
  reductions; corrected examples and error strings.
- **v0.1** — Initial release: covers authoring modes and the build-time / device boundary,
  data types and buffers, the datapath and streaming model, compute intrinsics, per-unit
  contracts, loop lowering, tiling, sync and fusion, high-level model authoring, and
  correctness hazards and limits.
