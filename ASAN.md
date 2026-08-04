# Running with AddressSanitizer and leak detection

`CFG=asan` enables both address sanitizer (memory errors) and leak detection via ASAN's built-in
LSan. A single build covers both inference.py and unit tests.

## Build

```bash
make operators CFG=asan   # builds fix_deepbind.so into operators/lib/ automatically
```

## Run unit tests

Unit tests run automatically as part of `make operators CFG=asan` — no separate step needed.
They run via ctest with the same ASAN/LSAN environment as inference.py. Results appear inline
during the build; a test failure aborts the build.

To run the tests without rebuilding:

```bash
cd operators/Asan
LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH LD_PRELOAD="$(pwd)/fix_deepbind.so $(gcc -print-file-name=libasan.so) $(g++ -print-file-name=libstdc++.so.6)" \
ASAN_OPTIONS=protect_shadow_gap=0:intercept_cxa_throw=0:verify_asan_link_order=0:new_delete_type_mismatch=0 \
LSAN_OPTIONS=suppressions=$(pwd)/../lsan.supp \
ctest --output-on-failure
```

## Run inference.py

```bash
LD_PRELOAD="$(pwd)/operators/lib/fix_deepbind.so /lib/x86_64-linux-gnu/libasan.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6" \
ASAN_OPTIONS=protect_shadow_gap=0:intercept_cxa_throw=0:verify_asan_link_order=0:new_delete_type_mismatch=0 \
LSAN_OPTIONS=suppressions=$(pwd)/operators/lsan.supp \
PYTHONMALLOC=malloc \
python inference.py <model> <input> --no-display
```

Memory errors are reported immediately. Leaks are reported at process exit.
`operators/lsan.supp` suppresses known false positives from Python, GStreamer, onnxruntime, and
Intel OpenCL — add entries there if new external false positives appear.

## Finding the right libasan and libstdc++ versions

Both versions must match whatever compiler built the operators. Ask the built library directly:

```bash
ldd operators/lib/libaxstreamer.so | grep -E "libasan|libstdc\+\+"
# e.g. libasan.so.6   => /lib/x86_64-linux-gnu/libasan.so.6        (GCC 11)
#      libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

Use those exact paths in `LD_PRELOAD`. Mismatching (e.g. preloading `libasan.so.8` when operators
link `libasan.so.6`) loads two ASAN runtimes simultaneously and causes spurious errors.

## Why each piece is needed

**LD_PRELOAD order matters:**

| Library | Reason |
|---|---|
| `fix_deepbind.so` | Strips `RTLD_DEEPBIND` from `dlopen` calls. Intel's OpenCL driver loads `libigfxdbgxchg64.so` with `RTLD_DEEPBIND`, which ASAN aborts on. Also provides a STRONG `sigaction` to bypass ASAN's broken interceptor (GCC 11's `REAL(sigaction)` lookup fails when the AIPU runtime is active). |
| `libasan.so.6` | Must be preloaded so shadow memory covers the full address space from process start. Without preloading, libasan loads late (when GStreamer opens the plugins) and shadow memory is broken. |
| `libstdc++.so.6` | Python does not link libstdc++, so at ASAN init time `dlsym(RTLD_NEXT, "__cxa_throw")` returns NULL. When onnxruntime calls `__cxa_throw`, ASAN's interceptor crashes. Preloading libstdc++ ensures it is present when ASAN initialises. |

**ASAN_OPTIONS:**

| Flag | Suppresses |
|---|---|
| `protect_shadow_gap=0` | Python startup crash: Python's `mmap` calls land in ASAN's shadow gap |
| `intercept_cxa_throw=0` | onnxruntime exception handling conflict with ASAN's `__cxa_throw` interceptor |
| `verify_asan_link_order=0` | Warning from `fix_deepbind.so` preceding libasan in the preload list |
| `new_delete_type_mismatch=0` | False positive from Intel GPU Compiler (`libigc.so.1`) during OpenCL kernel compilation |

**`PYTHONMALLOC=malloc`** — routes Python's internal allocations through system malloc so ASAN can track them.

## Restoring non-ASAN operators

```bash
make operators   # rebuilds with CFG=release and reinstalls to operators/lib/
```
