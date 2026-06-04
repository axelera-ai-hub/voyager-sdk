# Verify Your Setup

Run these checks to confirm your hardware, driver, and SDK are working correctly.

> [!CAUTION]
> **Before you start**
> Make sure you have activated your environment:
> ```bash
> source venv/bin/activate
> ```


## Step 1: Check that the system sees the hardware

```bash
lspci | grep Axelera
```

Expected output (example for a single PCIe card):

```
01:00.0 Processing accelerators: Axelera AI, Inc. Metis (rev 01)
```

If no output appears:
- Verify the card is physically seated correctly
- Check that the power supply meets the requirements
- Restart the system and try again

## Step 2: Check the device with [`axdevice`](../reference/tools/axdevice.md)

```bash
axdevice
```

Expected output (format varies by SDK version):

```
Device 0: metis-0:1:0 4GiB pcie flver=1.3.0 bcver=1.4 clock=800MHz(0-3:800MHz) mvm=0-3:100%
```

The key information: device index, name (PCIe bus address), memory, connection type, firmware version (`flver`), and clock speed. If multiple devices are installed, each appears on its own line.

> [!TIP]
> If `axdevice` is not found, your environment is not activated. Run `source venv/bin/activate` and try again.


## Step 3: Run a test inference

This runs a quick object detection on a sample video to verify the full stack — driver, runtime, compiler, and device. [`inference.py`](../reference/tools/inference-py.md) is the SDK's main command-line tool for running models:

```bash
./inference.py yolov5s-v7-coco media/traffic1_1080p.mp4 --no-display
```

> [!NOTE]
> The first run compiles the model for your hardware. This takes a few minutes and shows a progress bar. Subsequent runs start immediately.
>
> To skip compilation, you can pre-download a compiled model:
> ```bash
> axdownloadmodel yolov5s-v7-coco
> ```


Expected output:

```
[INFO] Building pipeline...
[INFO] Compiling model yolov5s-v7-coco...
████████████████████████████████████████ 100%
[INFO] Running inference...
[INFO] Processed 900 frames
[INFO] System throughput: XX.X FPS
[INFO] Device throughput: XX.X FPS
```

If this completes without errors, your setup is fully working.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `lspci` shows nothing | Card not seated or powered | Reseat card, check power, reboot |
| `axdevice` not found | Environment not activated | `source venv/bin/activate` |
| `axdevice` finds 0 devices | Driver not loaded | Re-run `./install.sh --all` |
| Inference fails at compilation | SDK/firmware mismatch | Check versions match with `axdevice` |
| Permission denied errors | Need sudo for driver access | Check group membership or run installer again |

---

## Setup complete

Your Metis hardware is installed, the SDK is working, and inference runs successfully.

Proceed to [First Inference](../user-guides/first-inference.md) to start working with models.
