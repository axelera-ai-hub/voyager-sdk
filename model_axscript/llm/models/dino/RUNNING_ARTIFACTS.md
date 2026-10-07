# Running a cached DINOv2 artifact on Metis (standalone)

This shows how to run a compiled DINOv2-base artifact given **only the build directory** — no
repo source needed beyond the installed `axelera.runtime`. It's a fully int8, single-dispatch
ViT: one graph call turns pixels into `last_hidden_state` (space-to-depth patch embed + 12
transformer blocks + final LayerNorm, all on device).

## What you have

A build dir, e.g. `dinov2-base-<hash>/`, containing:
- `model.json` — the graph spec (input/output names, dtypes, shapes). One graph: `encode`.
- `dinov2.elf`, `_embed/ _block/ _finalnorm/`, `outlined_weights__graph_input/` — the compiled
  kernels + baked int8 weights (loaded automatically via `model.json`).
- `meta.json` — the few host-side scalars a run needs:
  ```json
  {"repo_id": null, "image_size": 224, "patch_size": 14, "seq_real": 257, "seq_pad": 320,
   "hidden": 768, "grid": 16, "pix_scale": 0.020787402987480164, "mask_shape": [1,1,320,5,64]}
  ```

For reference, `model.json`'s `encode` graph (base @224) is:
- inputs:  `pixels` int8 `(1, num_patches, k_pw, 64)` = `(1,256,10,64)`, `attn_mask` int8 `(1,1,320,5,64)`
- outputs: `output` float32 `(1,320,12,64)`  (= `(1, seq_pad, hidden/64, 64)`)

`num_patches = grid*grid` (256), and `k_pw = ceil(C*P*P / 64)` is the per-patch im2col length in
64-lane blocks (`C*P*P = 3*14*14 = 588` → `k_pw = 10`). The embed is a space-to-depth **matmul**
(each patch flattened into the reduction dim), not a padded Conv2d.

## What you need

- A **Metis device** with ≥ `cores` free sub-devices (`axdevice` to check;
  `axdevice --reload-firmware` to recover leaked sub-devices — **never `--reboot`**).
- Python with `axelera.runtime` importable, plus `numpy`. If it won't import
  (`ModuleNotFoundError: axelera.config`, runtime bindings None), `source axelera-activate.sh`.
- For preprocessing: `pillow`, and ideally `transformers` (gives the *exact* eval transform).
  The model weights are NOT needed — they're baked into the artifact.

## The three host-side steps (what the device does NOT do)

1. **Preprocess** the image to the model's eval transform → a `(1,3,H,W)` float tensor
   (resize shortest-edge→256 bicubic, center-crop `image_size`, /255, ImageNet-normalize).
2. **im2col + int8-quantize** the pixels into the embed matmul's layout
   `(1, num_patches, k_pw, 64)`: flatten each `P×P` patch (in `c, kh, kw` order — the patch
   weight's K order) into the reduction dim, symmetric int8 quantize with `pix_scale`
   (zero-point 0), and zero-pad K up to a 64-lane multiple.
3. **Build the padding mask** `(1,1,seq_pad,ctx_blocks,64)` int8: `0` for real keys, `-127` for
   padded keys (keys ≥ `seq_real` are padding, since 257 real tokens are padded up to 320).

Then run the graph and slice the output back to `seq_real`.

## Full standalone script

```python
import json, numpy as np
from math import ceil
from pathlib import Path
from PIL import Image

ART   = Path("/path/to/dinov2-base-<hash>")   # <-- your build dir
CORES = 4                                       # must match how it was built
IMG   = "my_image.jpg"

spec = json.loads((ART / "model.json").read_text())
meta = json.loads((ART / "meta.json").read_text())
_NP = {"int8":np.int8,"int16":np.int16,"int32":np.int32,
       "float16":np.float16,"float32":np.float32,"uint64":np.uint64}

# --- 1. preprocess -> (1,3,H,W) float (exact HF eval transform) -----------------
from transformers import AutoImageProcessor
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
pixel_values = proc(images=Image.open(IMG).convert("RGB"),
                    return_tensors="np").pixel_values          # (1,3,H,W) float32
# (no transformers? see "Manual preprocessing" below)

# --- 2. im2col + int8 quantize into the embed matmul layout (1, num_patches, k_pw, 64) ---
px  = pixel_values[0]                                          # (C,H,W)
c, h, w = px.shape
p   = meta["patch_size"]
gh, gw = h // p, w // p
pix_i8 = np.clip(np.round(px / meta["pix_scale"]), -127, 127).astype(np.int8)
# (C, gh, kh, gw, kw) -> (gh, gw, C, kh, kw) -> (num_patches, C*P*P), flatten order (c, kh, kw)
cols = pix_i8.reshape(c, gh, p, gw, p).transpose(1, 3, 0, 2, 4).reshape(gh * gw, c * p * p)
k    = c * p * p
k_pw = ceil(k / 64)
data = np.zeros((1, gh * gw, k_pw * 64), np.int8)
data[0, :, :k] = cols                                          # remaining K lanes stay zero
pixels = data.reshape(1, gh * gw, k_pw, 64)

# --- 3. padding mask (1,1,seq_pad,ctx_blocks,64): 0 real, -127 pad ---------------
ms = meta["mask_shape"]; ctx_blocks = ms[-2]
keys = np.arange(ctx_blocks * 64).reshape(ctx_blocks, 64)
pad  = np.where(keys < meta["seq_real"], 0, -127).astype(np.int8)
attn_mask = np.broadcast_to(pad, ms).copy()

# --- load the artifact onto the device and run ----------------------------------
from axelera.runtime import Context
ctx = Context()
devices = ctx.list_devices()
if not devices:
    raise RuntimeError("no Axelera device/simulator available")
conn  = ctx.device_connect(device=devices[0], num_sub_devices=CORES)
model = ctx.load_model(str(ART / "model.json"))
inst  = conn.load_model_instance(model)

try:
    g = spec["graphs"]["encode"]
    named = {"pixels": pixels, "attn_mask": attn_mask}
    ordered = [np.ascontiguousarray(named[i["name"]], dtype=_NP[i["dtype"]]) for i in g["inputs"]]
    o = g["outputs"][0]
    out = np.zeros(tuple(o["shape"]), dtype=_NP[o["dtype"]])   # (1,320,12,64) float32
    inst._run_graph(graph="encode", inputs=ordered, outputs=[out])

    # reshape (1,seq_pad,hidden) and drop padding -> HF last_hidden_state
    hidden = out.reshape(1, meta["seq_pad"], meta["hidden"])[:, : meta["seq_real"]]  # (1,257,768)
    cls_token    = hidden[0, 0]        # (768,) global image embedding
    patch_tokens = hidden[0, 1:]       # (256,768) = 16x16 patch grid (meta["grid"]==16)
    print(cls_token.shape, patch_tokens.shape)
finally:
    ctx.release()                      # ALWAYS release, or you leak sub-devices
```

For many images: build `pixels`/run/reshape in a loop, reusing the same `inst` (one dispatch
per image, ~62 ms / ~16 FPS on 4 cores @224); release the context once at the end.

## Output semantics

`hidden` equals HF `Dinov2Model(pixel_values).last_hidden_state[:, :257]`, `(1,257,768)`
float32: token 0 is the CLS (image-level) embedding, tokens 1..256 are the 16×16 patch grid
(row-major). Fidelity vs fp on this build: CLS-cosine 0.964, retrieval recall@20 0.89.

## Manual preprocessing (no `transformers`)

Must match the training transform exactly, or features degrade. DINOv2 uses: resize the
shortest edge to 256 (bicubic), center-crop `image_size` (224), scale to [0,1], then normalize
with ImageNet mean `(0.485,0.456,0.406)` / std `(0.229,0.224,0.225)`, channels-first:

```python
from PIL import Image
img = Image.open(IMG).convert("RGB")
w, h = img.size; s = 256 / min(w, h)
img = img.resize((round(w*s), round(h*s)), Image.BICUBIC)
w, h = img.size; l, t = (w-224)//2, (h-224)//2
img = img.crop((l, t, l+224, t+224))
x = np.asarray(img, np.float32) / 255.0                    # (224,224,3)
x = (x - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
pixel_values = np.transpose(x, (2,0,1))[None]              # (1,3,224,224)
```

## Gotchas

- **`num_sub_devices` must match the build** (this artifact is 4-core).
- **The patch layout is im2col, not a conv.** Pixels go in as `(1, num_patches, k_pw, 64)` with
  each patch flattened in `(c, kh, kw)` order and K zero-padded to 64 lanes. Artifacts built
  before this change used a `(1,H,W,1,64)` zero-padded-conv layout — rebuild if in doubt.
- All compute is on device; the host does only preprocess → im2col/int8 quantize → run → slice.
- Inputs must be C-contiguous and the exact dtype from `model.json` (the script casts them).
- The output is pre-allocated by the caller and filled in place by `_run_graph`.
```
