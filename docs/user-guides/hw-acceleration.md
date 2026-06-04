---
title: "Host Hardware Acceleration with OpenCL and VA-API"
---

# Host Hardware Acceleration with OpenCL and VA-API

## Table of Contents

1. [Overview](#1-overview)
2. [How the SDK Uses OpenCL and VA-API](#2-how-the-sdk-uses-opencl-and-va-api)
3. [Platform Summary](#3-platform-summary)
4. [Intel Host: Prerequisites and Installation](#4-intel-host-prerequisites-and-installation)
   - [4.1 System Requirements](#41-system-requirements)
   - [4.2 Required Packages — Intel](#42-required-packages--intel)
   - [4.3 Intel GPU Generation Compatibility Matrix](#43-intel-gpu-generation-compatibility-matrix)
5. [AMD Ryzen APU Host: Prerequisites and Installation](#5-amd-ryzen-apu-host-prerequisites-and-installation)
   - [5.1 System Requirements](#51-system-requirements)
   - [5.2 Required Packages — AMD](#52-required-packages--amd)
   - [5.3 OpenCL Runtime Options for AMD APU](#53-opencl-runtime-options-for-amd-apu)
   - [5.4 AMD APU Generation Compatibility Matrix](#54-amd-apu-generation-compatibility-matrix)
6. [Common Setup: User Group Membership](#6-common-setup-user-group-membership)
7. [GStreamer VA-API Setup](#7-gstreamer-va-api-setup)
   - [7.1 Required Packages](#71-required-packages)
   - [7.2 GStreamer VA-API Elements Reference](#72-gstreamer-va-api-elements-reference)
   - [7.3 Ensuring vaapih264dec Is Selected by decodebin](#73-ensuring-vaapih264dec-is-selected-by-decodebin)
8. [Automatic Detection Behaviour](#8-automatic-detection-behaviour)
9. [Controlling Acceleration: CLI Flags](#9-controlling-acceleration-cli-flags)
   - [9.1 inference.py / deploy.py Flags](#91-inferencepy--deploypy-flags)
   - [9.2 Common Usage Patterns by Platform](#92-common-usage-patterns-by-platform)
10. [Controlling Acceleration: Python API](#10-controlling-acceleration-python-api)
11. [Verification](#11-verification)
    - [11.1 Verify GPU Device Node](#111-verify-gpu-device-node)
    - [11.2 Verify OpenCL Is Available](#112-verify-opencl-is-available)
    - [11.3 Verify VA-API Is Available](#113-verify-va-api-is-available)
    - [11.4 Verify GStreamer VA-API Element Selection](#114-verify-gstreamer-va-api-element-selection)
    - [11.5 Confirm Acceleration Is Active at Runtime](#115-confirm-acceleration-is-active-at-runtime)
    - [11.6 Monitor GPU Utilisation](#116-monitor-gpu-utilisation)
12. [Important Notes and Caveats](#12-important-notes-and-caveats)
    - [12.1 VA-API Detection Is Vendor-Agnostic](#121-va-api-detection-is-vendor-agnostic)
    - [12.2 GStreamer Decoder Rank Conflict on Ubuntu 22.04](#122-gstreamer-decoder-rank-conflict-on-ubuntu-2204)
    - [12.3 OpenCL Platform Conflicts (Multi-GPU / Mixed-Vendor Systems)](#123-opencl-platform-conflicts-multi-gpu--mixed-vendor-systems)
    - [12.4 Kernel Version Requirements](#124-kernel-version-requirements)
    - [12.5 ROCm Does Not Officially Support AMD APU iGPUs](#125-rocm-does-not-officially-support-amd-apu-igpus)
    - [12.6 Mesa RustiCL vs. Clover](#126-mesa-rusticl-vs-clover)
    - [12.7 AMDGPU Power Profile for OpenCL Workloads](#127-amdgpu-power-profile-for-opencl-workloads)
13. [Troubleshooting](#13-troubleshooting)
    - [13.1 Intel-specific Issues](#131-intel-specific-issues)
    - [13.2 AMD-specific Issues](#132-amd-specific-issues)
    - [13.3 Common Issues (Both Platforms)](#133-common-issues-both-platforms)
14. [Quick Reference](#14-quick-reference)

---

## 1. Overview

The Axelera Voyager SDK builds end-to-end AI inference pipelines running on the **Metis AIPU** (PCIe / M.2) together with host-side pre-processing and post-processing. To maximise throughput and minimise CPU load, the SDK offloads two categories of host-side work to dedicated hardware:

| Accelerator | Host Hardware                                                              | Role in the Pipeline                                                                             |
| ----------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **VA-API**  | Intel iGPU (Quick Sync / `iHD`) · AMD Radeon iGPU (`radeonsi` Mesa driver) | Hardware video **decode**: H.264 / H.265 / AV1 / VP9 without consuming CPU cycles                |
| **OpenCL**  | Intel iGPU / Arc dGPU · AMD Radeon iGPU (amdgpu)                           | GPU-accelerated image **pre-processing**: resize, colour conversion, normalisation, letterboxing |

---

## 2. How the SDK Uses OpenCL and VA-API

```
[Camera / Video File]
        │
        ▼ VA-API (Intel iHD  ─or─  AMD radeonsi via Mesa mesa-va-drivers package)
[HW Video Decode]
        │  (dmabuf)
        ▼ OpenCL (Intel GPU  ─or─  AMD Radeon iGPU)
[Image Pre-Processing]    ← AxTransform (OpenCL kernels)
   resize · letterbox · colour convert · normalise
        │  (dmabuf)
        ▼
[Metis AIPU Inference]    ← AxInference (Metis PCIe / M.2)
        │
        ▼ OpenCL (optional post-processing)
[Application Output]
```

During **pipeline compilation** (`deploy.py`), the SDK inspects available hardware accelerators and compiles OpenCL kernels targeting the detected GPU. Compiled pipeline artifacts are device-specific — changing the hardware configuration requires recompilation.

---

## 3. Platform Summary

| Feature                 | Intel Core / Core Ultra               | AMD Ryzen APU                                                |
| ----------------------- | ------------------------------------- | ------------------------------------------------------------ |
| VA-API (Voyager SDK)    | **Supported** (`iHD` / `i965` driver) | **Supported** (`radeonsi` driver, `mesa-va-drivers` package) |
| OpenCL                  | Supported (Intel NEO runtime)         | Supported (Mesa RustiCL or ROCm)                             |
| GPU kernel driver       | `i915` / `xe`                         | `amdgpu`                                                     |
| VA-API driver name      | `iHD` (Gen8+) / `i965` (Gen6–7)       | `radeonsi`                                                   |
| OpenCL ICD              | `intel.icd`                           | `mesa.icd` (RustiCL) or AMD APP (ROCm)                       |
| Recommended OpenCL path | `intel-opencl-icd` (NEO)              | `mesa-opencl-icd` + `RUSTICL_ENABLE=radeonsi`                |
| `render` group required | Yes                                   | Yes                                                          |
| `video` group required  | Yes                                   | Yes                                                          |

---

## 4. Intel Host: Prerequisites and Installation

### 4.1 System Requirements

- Intel Core (6th–13th Gen) or Core Ultra (1st–2nd Gen) with integrated GPU
- `/dev/dri/renderD128` must exist (confirms i915 / xe driver is loaded)
- Kernel:
  - Ubuntu 22.04: ≥ 5.15 standard; ≥ 6.2 for Core Ultra / Meteor Lake (`linux-generic-hwe-22.04`)
  - Ubuntu 24.04: ≥ 6.8 standard; ≥ 6.12 for Lunar Lake Xe2 (`linux-generic-hwe-24.04`)

### 4.2 Required Packages — Intel

```bash
# Option A: Ubuntu default repository (6th–11th Gen, sufficient for most use cases)
sudo apt update
sudo apt install -y \
  intel-opencl-icd ocl-icd-opencl-dev opencl-headers clinfo \
  libva2 libva-drm2 libva-x11-2 libva-wayland2 \
  intel-media-va-driver-non-free \
  vainfo intel-gpu-tools

# Option B: Intel graphics repository (recommended for 12th Gen and later)
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg

# Ubuntu 22.04
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
  https://repositories.intel.com/gpu/ubuntu jammy client" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

# Ubuntu 24.04
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
  https://repositories.intel.com/gpu/ubuntu noble client" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list

sudo apt update && sudo apt install -y \
  intel-opencl-icd intel-level-zero-gpu level-zero \
  intel-media-va-driver-non-free libmfx1 libmfxgen1 \
  ocl-icd-opencl-dev clinfo vainfo intel-gpu-tools

# Legacy driver for 6th–7th Gen (if iHD does not initialise)
# sudo apt install -y i965-va-driver
```

Set VA-API driver name if not auto-detected:

```bash
# Gen8+ (Kaby Lake and later) — persistent
echo 'export LIBVA_DRIVER_NAME=iHD' >> ~/.bashrc && source ~/.bashrc

# Gen6–7 (Skylake / Kaby Lake early)
# echo 'export LIBVA_DRIVER_NAME=i965' >> ~/.bashrc && source ~/.bashrc
```

Install HWE kernel for Core Ultra on Ubuntu 22.04:

```bash
sudo apt install -y linux-generic-hwe-22.04
sudo reboot
```

### 4.3 Intel GPU Generation Compatibility Matrix

| CPU Generation                      | GPU Architecture     | OpenCL    | VA-API Driver  | AV1 Dec HW | AV1 Enc HW | Key Notes                                    |
| ----------------------------------- | -------------------- | --------- | -------------- | :--------: | :--------: | -------------------------------------------- |
| 6th Gen (Skylake)                   | Gen9                 | 2.0 (NEO) | i965 / iHD     |     ✗      |     ✗      | HEVC decode only                             |
| 7th Gen (Kaby Lake)                 | Gen9.5               | 2.0 (NEO) | iHD            |     ✗      |     ✗      | HEVC encode added; VP9 decode                |
| 8th–9th Gen (Coffee Lake)           | Gen9.5               | 2.0 (NEO) | iHD            |     ✗      |     ✗      | VP9 decode                                   |
| 10th–11th Gen (Ice/Tiger Lake)      | Gen11/Gen12 Xe-LP    | 3.0       | iHD            |  Tiger: ✓  |     ✗      | AV1 decode from Tiger Lake                   |
| 12th–13th Gen (Alder/Raptor Lake)   | Gen12.2 Xe-LP        | 3.0       | iHD (non-free) |     ✓      |     ✗      | SR-IOV; HEVC 12bit                           |
| Core Ultra 1st Gen (Meteor Lake)    | Gen12.5 Xe-LPG       | 3.0       | iHD (non-free) |     ✓      |     ✓      | NPU 3.5 (10 TOPS); kernel ≥ 6.2              |
| Core Ultra 2nd Gen S/H (Arrow Lake) | Gen12.5 Xe-LPG       | 3.0       | iHD (non-free) |     ✓      |     ✓      | NPU 13 TOPS                                  |
| Core Ultra 2nd Gen V (Lunar Lake)   | Xe2-LPG (Battlemage) | 3.0       | iHD (non-free) |     ✓      |     ✓      | VVC/H.266 HW; NPU 4 (48 TOPS); kernel ≥ 6.12 |

---

## 5. AMD Ryzen APU Host: Prerequisites and Installation

### 5.1 System Requirements

- AMD Ryzen APU with integrated Radeon GPU (Raven Ridge / Picasso / Renoir / Cezanne / Rembrandt / Phoenix / Hawk Point / Strix Point or later)
- `/dev/dri/renderD128` must exist (confirms `amdgpu` kernel driver is loaded)
- Kernel:
  - Ubuntu 22.04: ≥ 5.15 for APUs up to Cezanne; ≥ 6.5 recommended for Rembrandt (Ryzen 6000) and later
  - Ubuntu 24.04: ≥ 6.8 for most RDNA 3 APUs; ≥ 6.10 for Strix Point (Ryzen AI 300)
- Mesa version:
  - Ubuntu 22.04 ships Mesa 22.x; upgrade via `kisak-mesa` PPA to get Mesa 24.x with full RustiCL and improved `radeonsi` VA-API support
  - Ubuntu 24.04 ships Mesa 24.x — adequate for RDNA 2 and RDNA 3 APUs out of the box

### 5.2 Required Packages — AMD

```bash
sudo apt update
sudo apt install -y \
  libva2 libva-drm2 libva-x11-2 libva-wayland2 \
  mesa-va-drivers \
  vainfo \
  mesa-opencl-icd clinfo ocl-icd-opencl-dev

# Optional: FFmpeg for hardware decode verification
sudo apt install -y ffmpeg
```

Set the VA-API driver name for AMD:

```bash
echo 'export LIBVA_DRIVER_NAME=radeonsi' >> ~/.bashrc
echo 'export LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri' >> ~/.bashrc
source ~/.bashrc
```

See [Section 5.3](#53-opencl-runtime-options-for-amd-apu) for OpenCL (RustiCL) setup.

### 5.3 OpenCL Runtime Options for AMD APU

Three OpenCL runtimes are available for AMD APU iGPUs. **Option 1 (Mesa RustiCL) is recommended** for all APU integrated graphics.

#### Option 1: Mesa RustiCL ✅ Recommended

RustiCL is a modern OpenCL 3.0 implementation included in Mesa 23.1+. It is the recommended path for AMD APU iGPUs because ROCm does not officially support APU integrated graphics, and it requires no additional repositories on Ubuntu 24.04.

**Ubuntu 24.04** (Mesa 24.x ships with RustiCL out of the box):

```bash
sudo apt install -y mesa-opencl-icd clinfo
echo 'export RUSTICL_ENABLE=radeonsi' >> ~/.bashrc && source ~/.bashrc
clinfo -l
# Expected: Platform: rusticl  Device: AMD Radeon Graphics (radeonsi, ...)
```

**Ubuntu 22.04** (Mesa 22.x does not include RustiCL — upgrade via kisak-mesa PPA):

```bash
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt update
sudo apt install -y mesa-opencl-icd clinfo
echo 'export RUSTICL_ENABLE=radeonsi' >> ~/.bashrc && source ~/.bashrc
clinfo -l
```

> **Clover vs. RustiCL:** Without `RUSTICL_ENABLE=radeonsi`, `mesa-opencl-icd` on Ubuntu 22.04 defaults to the legacy Clover driver (OpenCL 1.1), which lacks critical extensions required by the Voyager SDK's GPU kernels. Always confirm `clinfo` reports `Platform Name: rusticl` — not `Clover`.

#### Option 2: ROCm

The official AMD OpenCL stack (`AMD Accelerated Parallel Processing` platform), intended for supported **discrete** Radeon GPUs (GFX9 / RDNA and later). AMD does not officially support ROCm on APU integrated graphics. Use this option only when a supported discrete Radeon GPU is present alongside the APU. Refer to the [ROCm installation documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) for setup instructions.

#### Option 3: amdgpu-install (APU iGPU)

An alternative path using AMD's `amdgpu-install` tool with `--usecase=opencl --no-dkms`, which installs the OpenCL user-space stack without replacing the Ubuntu-supplied `amdgpu` kernel module. Use this only when Option 1 is not viable. Refer to the [AMD GPU install documentation](https://amdgpu-install.readthedocs.io/) for details.

### 5.4 AMD APU Generation Compatibility Matrix

| Ryzen Series               | Codename           | GPU Architecture | VCN     | OpenCL      | VA-API (OS) | AV1 Dec | AV1 Enc | Notes                                           |
| -------------------------- | ------------------ | ---------------- | ------- | ----------- | ----------- | :-----: | :-----: | ----------------------------------------------- |
| Ryzen 2000G (2018)         | Raven Ridge        | GCN 5 (Vega)     | VCN 1.0 | RustiCL 3.0 | radeonsi    |    ✗    |    ✗    | VP9 decode (VCN 1.0)                            |
| Ryzen 3000G / 3000U        | Picasso            | GCN 5 (Vega)     | VCN 1.0 | RustiCL 3.0 | radeonsi    |    ✗    |    ✗    | Same as Raven Ridge                             |
| Ryzen 4000G / 4000U        | Renoir             | GCN 5 (Vega)     | VCN 2.2 | RustiCL 3.0 | radeonsi    |    ✗    |    ✗    | H.264/HEVC enc; VP9 dec                         |
| Ryzen 5000G / 5000U        | Cezanne / Lucienne | GCN 5 (Vega)     | VCN 2.x | RustiCL 3.0 | radeonsi    |    ✗    |    ✗    | Last GCN-based APU                              |
| Ryzen 6000U/H              | Rembrandt          | RDNA 2           | VCN 3.1 | RustiCL 3.0 | radeonsi    |    ✓    |    ✗    | First RDNA 2 APU; AV1 decode                    |
| Ryzen 7000G (desktop)      | Raphael / Phoenix  | RDNA 2 (I/O die) | VCN 3.x | RustiCL 3.0 | radeonsi    |    ✓    |    ✗    | Small iGPU on I/O die                           |
| Ryzen 7040 / 7045HX        | Phoenix            | RDNA 3           | VCN 4.0 | RustiCL 3.0 | radeonsi    |    ✓    |    ✓    | AV1 encode; XDNA NPU; Ryzen AI                  |
| Ryzen 8000G (desktop)      | Hawk Point         | RDNA 3           | VCN 4.0 | RustiCL 3.0 | radeonsi    |    ✓    |    ✓    | Radeon 760M–890M; XDNA NPU                      |
| Ryzen AI 300 (Strix Point) | Strix Point        | RDNA 3.5         | VCN 5.0 | RustiCL 3.0 | radeonsi    |    ✓    |    ✓    | XDNA 2 NPU (50 TOPS); kernel ≥ 6.10; Mesa 25.x+ |

---

## 6. Common Setup: User Group Membership

Both Intel and AMD hosts require the user running the SDK to belong to the `render` and `video` groups:

```bash
sudo usermod -a -G render,video $USER

# Apply without logout (current session only)
newgrp render

# Verify
groups $USER
# Expected output contains: ... render video ...
```

---

## 7. GStreamer VA-API Setup

The Voyager SDK's `axelera.app` module uses GStreamer as its default pipeline backend (`inference.py` defaults to `--pipe=gst`). VA-API hardware decode is provided to GStreamer through the `gstreamer1.0-vaapi` plugin package, which exposes `vaapih264dec`, `vaapipostproc`, and related elements. These must be installed and correctly configured for the SDK's VA-API acceleration to function end-to-end.

### 7.1 Required Packages

```bash
# GStreamer core and VA-API plugin
sudo apt install -y \
  gstreamer1.0-vaapi \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-libav

# VA-API driver (if not already installed)
# Intel (Gen8+):
sudo apt install -y intel-media-va-driver-non-free
# AMD:
sudo apt install -y mesa-va-drivers
```

> **Ubuntu 22.04 package version:** `gstreamer1.0-vaapi` is version `1.20.x` on Ubuntu 22.04 and `1.24.x` on Ubuntu 24.04. Both are available in the `universe` repository.
>
> **Known issue — Ubuntu 22.04 `1.20.1`:** A bug in `gstreamer1.0-vaapi 1.20.1` can cause H.264 decoding to produce a black frame output. If this is observed, upgrading GStreamer via the `kisak-mesa` PPA or using `--disable-vaapi` as a workaround is recommended.

### 7.2 GStreamer VA-API Elements Reference

Once `gstreamer1.0-vaapi` is installed, the following elements are available:

| Element          | Type           | Description                                   |
| ---------------- | -------------- | --------------------------------------------- |
| `vaapidecodebin` | Decoder (auto) | Auto-selecting VA-API decode bin              |
| `vaapih264dec`   | Decoder        | H.264 / AVC hardware decode                   |
| `vaapih265dec`   | Decoder        | H.265 / HEVC hardware decode                  |
| `vaapiav1dec`    | Decoder        | AV1 hardware decode                           |
| `vaapivp9dec`    | Decoder        | VP9 hardware decode                           |
| `vaapimpeg2dec`  | Decoder        | MPEG-2 hardware decode                        |
| `vaapih264enc`   | Encoder        | H.264 hardware encode                         |
| `vaapih265enc`   | Encoder        | H.265 hardware encode                         |
| `vaapijpegenc`   | Encoder        | JPEG hardware encode                          |
| `vaapipostproc`  | Filter         | Colour space conversion, scaling, deinterlace |
| `vaapivideosink` | Sink           | VA-API accelerated display output             |

Verify the plugin is loaded correctly:

```bash
gst-inspect-1.0 vaapi
# Expected: lists all vaapi* elements above

# Inspect a specific element
gst-inspect-1.0 vaapih264dec
gst-inspect-1.0 vaapipostproc
```

If `gst-inspect-1.0 vaapi` returns nothing, `gstreamer1.0-vaapi` is not installed or the VA-API driver failed to initialise. Verify `vainfo` succeeds first (see [Section 11.3](#113-verify-va-api-is-available)).

### 7.3 Ensuring vaapih264dec Is Selected by decodebin

The Voyager SDK uses `decodebin` internally to auto-select the appropriate decoder. GStreamer selects elements based on **rank** — higher rank wins. On Ubuntu 22.04, `vaapih264dec` and `avdec_h264` (FFmpeg software decoder) both have the same rank of `primary (256)`, which means the selection is non-deterministic and may fall back to the software decoder.

**Check current ranks:**

```bash
gst-inspect-1.0 vaapih264dec | grep -i rank
# Ubuntu 22.04: Rank  primary (256)   ← same as avdec_h264

gst-inspect-1.0 avdec_h264 | grep -i rank
# Rank  primary (256)

gst-inspect-1.0 openh264dec | grep -i rank
# Rank  marginal (64)   ← already lower, not a concern
```

**Solution — raise `vaapih264dec` rank above `avdec_h264`:**

Set the `GST_PLUGIN_FEATURE_RANK` environment variable before launching the SDK:

```bash
# Raise vaapih264dec (and other VA-API decoders) above primary (256)
export GST_PLUGIN_FEATURE_RANK=vaapih264dec:257,vaapih265dec:257,vaapiav1dec:257,vaapivp9dec:257
```

Make this persistent:

```bash
echo 'export GST_PLUGIN_FEATURE_RANK=vaapih264dec:257,vaapih265dec:257,vaapiav1dec:257,vaapivp9dec:257' \
  >> ~/.bashrc && source ~/.bashrc
```

Verify that the rank override has taken effect:

```bash
GST_PLUGIN_FEATURE_RANK=vaapih264dec:257 \
  gst-inspect-1.0 vaapih264dec | grep -i rank
# Expected: Rank  primary+1 (257)
```

> **Ubuntu 24.04:** `gstreamer1.0-vaapi 1.24.x` may assign a higher default rank to VA-API elements depending on driver availability at plugin registration time. Verify with `gst-inspect-1.0 vaapih264dec | grep -i rank` and apply `GST_PLUGIN_FEATURE_RANK` if still at 256.

---

## 8. Automatic Detection Behaviour

By default, the Voyager SDK automatically detects available hardware accelerators at two points:

1. **At pipeline compilation time** (`deploy.py`): the compiler scans for OpenCL and VA-API and compiles kernels for the best available hardware.
2. **At inference time** (`inference.py` / `create_inference_stream`): the runtime re-checks accelerator availability before running the pipeline.

Detection probes:

- **OpenCL:** queries `clGetPlatformIDs` for available platforms
- **VA-API:** probes `/dev/dri/renderD128` and attempts VA driver initialisation — any driver that initialises successfully (Intel `iHD`, Intel `i965`, AMD `radeonsi`, …) is accepted; there is no vendor-specific filter

If a required accelerator is found, it is used automatically. If not found, the SDK falls back to CPU-based processing (higher CPU utilisation, lower throughput).

---

## 9. Controlling Acceleration: CLI Flags

### 9.1 inference.py / deploy.py Flags

| Flag                      | Effect                                                            |
| ------------------------- | ----------------------------------------------------------------- |
| `--auto-vaapi`            | Auto-detect VA-API _(default)_                                    |
| `--enable-vaapi`          | Force-enable VA-API even if auto-detection gives a false negative |
| `--disable-vaapi`         | Disable VA-API; fall back to CPU video decode                     |
| `--auto-opencl`           | Auto-detect OpenCL _(default)_                                    |
| `--enable-opencl`         | Force-enable OpenCL if auto-detection gives a false negative      |
| `--disable-opencl`        | Disable OpenCL; fall back to CPU pre-processing                   |
| `--auto-opengl`           | Auto-detect OpenGL for annotation rendering _(default)_           |
| `--enable-opengl`         | Force-enable OpenGL rendering                                     |
| `--disable-opengl`        | Use OpenCV for annotation rendering                               |
| `--enable-hardware-codec` | Enable GStreamer hardware codec elements                          |

### 9.2 Common Usage Patterns by Platform

**Intel host — recommended default (both VA-API and OpenCL auto-detected):**

```bash
./inference.py yolov8s-coco media/video.mp4
```

**Intel host — force-enable both (use only if auto-detect gives a false negative):**

```bash
./inference.py yolov8s-coco media/video.mp4 --enable-vaapi --enable-opencl
```

**AMD Ryzen APU host — recommended default (VA-API and OpenCL both auto-detected):**

```bash
./inference.py yolov8s-coco media/video.mp4
```

The SDK will detect the `radeonsi` VA-API driver via the `mesa-va-drivers` package automatically. Ensure `LIBVA_DRIVER_NAME=radeonsi` and `RUSTICL_ENABLE=radeonsi` are exported before launching.

**AMD Ryzen APU host — force-enable both (false negative workaround):**

```bash
./inference.py yolov8s-coco media/video.mp4 --enable-vaapi --enable-opencl
```

**AMD Ryzen APU host — OpenCL only (VA-API unavailable or problematic):**

```bash
./inference.py yolov8s-coco media/video.mp4 --disable-vaapi
```

**AMD Ryzen APU host — CPU fallback (no GPU acceleration):**

```bash
./inference.py yolov8s-coco media/video.mp4 --disable-vaapi --disable-opencl
```

**deploy.py with explicit platform flags:**

```bash
# Intel
./deploy.py --enable-vaapi --enable-opencl my_network.yaml

# AMD (auto-detect recommended; explicit if needed)
./deploy.py --auto-vaapi --auto-opencl my_network.yaml
```

---

## 10. Controlling Acceleration: Python API

```python
from axelera.app import config
from axelera.app.stream import create_inference_stream

# Intel host — all auto-detect (recommended)
stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=["media/traffic1_1080p.mp4"],
    pipe_type="gst",
    hardware_caps=config.HardwareCaps(
        vaapi=config.HardwareEnable.detect,    # Intel iHD driver
        opencl=config.HardwareEnable.detect,   # Intel GPU kernels
        opengl=config.HardwareEnable.detect,
    ),
)

# AMD Ryzen APU host — all auto-detect (recommended)
# Requires: LIBVA_DRIVER_NAME=radeonsi and RUSTICL_ENABLE=radeonsi in environment
stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=["media/traffic1_1080p.mp4"],
    pipe_type="gst",
    hardware_caps=config.HardwareCaps(
        vaapi=config.HardwareEnable.detect,    # AMD radeonsi (mesa-va-drivers)
        opencl=config.HardwareEnable.detect,   # AMD Radeon via RustiCL
        opengl=config.HardwareEnable.detect,
    ),
)

# AMD Ryzen APU host — OpenCL only (VA-API unavailable or problematic)
stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=["media/traffic1_1080p.mp4"],
    pipe_type="gst",
    hardware_caps=config.HardwareCaps(
        vaapi=config.HardwareEnable.disable,
        opencl=config.HardwareEnable.detect,
        opengl=config.HardwareEnable.detect,
    ),
)

# Any host — CPU fallback (no GPU acceleration)
stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=["media/traffic1_1080p.mp4"],
    pipe_type="gst",
    hardware_caps=config.HardwareCaps(
        vaapi=config.HardwareEnable.disable,
        opencl=config.HardwareEnable.disable,
        opengl=config.HardwareEnable.disable,
    ),
)
```

`HardwareEnable` values:

| Value                           | Behaviour                               |
| ------------------------------- | --------------------------------------- |
| `config.HardwareEnable.detect`  | Auto-detect _(default)_                 |
| `config.HardwareEnable.enable`  | Force-enable (only for false negatives) |
| `config.HardwareEnable.disable` | Disable completely                      |

---

## 11. Verification

### 11.1 Verify GPU Device Node

```bash
ls -la /dev/dri/
# Expected: renderD128 and card0 (or card1) should be present

# Identify kernel driver
lsmod | grep -E "i915|amdgpu"
dmesg | grep -i "i915\|amdgpu\|drm" | tail -20
```

### 11.2 Verify OpenCL Is Available

```bash
clinfo -l
```

Expected — **Intel host (NEO runtime):**

```
Platform #0: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) UHD Graphics 770
```

Expected — **AMD host (RustiCL):**

```
Platform #0: rusticl
 `-- Device #0: AMD Radeon Graphics (radeonsi, gfx1036, LLVM 17.x, ...)
```

Expected — **AMD host (ROCm):**

```
Platform #0: AMD Accelerated Parallel Processing
 `-- Device #0: gfx1036
```

> If `clinfo` shows `Platform Name: Clover` with `OpenCL 1.1`, the legacy Clover driver is active. Enable RustiCL with `export RUSTICL_ENABLE=radeonsi` or upgrade Mesa.

Additional detail:

```bash
clinfo | grep -E "Platform Name|Device Name|Device Version|Max compute units"
```

### 11.3 Verify VA-API Is Available

**Intel host:**

```bash
vainfo --display drm --device /dev/dri/renderD128
# Expected driver: "Intel iHD driver for Intel(R) Gen Graphics"
```

**AMD host:**

```bash
LIBVA_DRIVER_NAME=radeonsi vainfo --display drm --device /dev/dri/renderD128
# Expected: "Mesa Gallium driver ... for AMD Radeon Graphics (radeonsi, ...)"
```

Key VA profiles to verify:

- `VAProfileH264Main : VAEntrypointVLD` — H.264 hardware decode
- `VAProfileHEVCMain : VAEntrypointVLD` — HEVC hardware decode
- `VAProfileAV1Profile0 : VAEntrypointVLD` — AV1 hardware decode (RDNA 2+ / Tiger Lake+)
- `VAEntrypointEncSlice` — hardware encode support

### 11.4 Verify GStreamer VA-API Element Selection

The Voyager SDK uses `decodebin` internally via the GStreamer backend. Before running the SDK, confirm that `decodebin` selects the VA-API decoder rather than the software fallback.

**Step 1 — Generate a test clip:**

```bash
# YUV420p H.264 FHD 5-second test clip using FFmpeg built-in test pattern
ffmpeg -f lavfi -i testsrc=size=1920x1080:rate=30 \
  -t 5 -c:v libx264 -pix_fmt yuv420p -preset fast -crf 23 \
  test_fhd_h264.mp4
```

**Step 2 — Check ranks and apply override if needed:**

```bash
gst-inspect-1.0 vaapih264dec | grep -i rank
gst-inspect-1.0 avdec_h264   | grep -i rank
# If both show primary (256), set the override:
export GST_PLUGIN_FEATURE_RANK=vaapih264dec:257,vaapih265dec:257,vaapiav1dec:257,vaapivp9dec:257
```

**Step 3 — Confirm decodebin selects vaapih264dec via debug log:**

```bash
GST_DEBUG=decodebin:5 \
gst-launch-1.0 filesrc location=test_fhd_h264.mp4 ! \
  decodebin ! fakesink sync=false 2>&1 | grep -iE "vaapi|avdec|selected|adding"
# Expected: vaapih264dec appears; avdec_h264 does not
```

**Step 4 — Smoke test with explicit VA-API pipeline:**

```bash
# Intel
gst-launch-1.0 filesrc location=test_fhd_h264.mp4 ! \
  qtdemux ! h264parse ! vaapih264dec ! vaapipostproc ! fakesink sync=false

# AMD
LIBVA_DRIVER_NAME=radeonsi \
gst-launch-1.0 filesrc location=test_fhd_h264.mp4 ! \
  qtdemux ! h264parse ! vaapih264dec ! vaapipostproc ! fakesink sync=false
```

A clean run with no errors confirms the full VA-API decode path is functional.

**Step 5 — Confirm GPU activity during decode:**

Run the GPU monitoring tool for your platform (see [Section 11.6](#116-monitor-gpu-utilisation)) in a separate terminal while Step 4 is executing. The Video / Decode utilisation should rise above the idle baseline. If it stays at 0%, the software decoder is being selected — apply the rank override in Step 2 and retry. See also [Section 12.2](#122-gstreamer-decoder-rank-conflict-on-ubuntu-2204).

### 11.5 Confirm Acceleration Is Active at Runtime

```bash
./inference.py --loglevel INFO yolov8s-coco media/video.mp4 2>&1 | \
  grep -i "vaapi\|opencl\|accel\|WARNING"
```

- Acceleration active: `Using VAAPI` or `OpenCL platform detected` in log output.
- Acceleration absent / disabled: `WARNING: No OpenCL GPU devices found` or similar.

In the Python API, pass `log_level=logging_utils.DEBUG` to `create_inference_stream`.

### 11.6 Monitor GPU Utilisation

**Intel host:**

```bash
sudo intel_gpu_top
# During active inference: Video row (VA-API decode) and Render/Compute row
# (OpenCL pre-processing) should show non-zero percentages.
```

**AMD host:**

```bash
# Option 1: radeontop
sudo apt install -y radeontop
radeontop

# Option 2: sysfs GPU busy percent
watch -n 1 cat /sys/class/drm/card0/device/gpu_busy_percent

# Option 3: amdgpu_top (available with ROCm installation)
amdgpu_top
```

During active OpenCL inference, the GPU busy percentage should be elevated above the idle baseline.

---

## 12. Important Notes and Caveats

### 12.1 VA-API Detection Is Vendor-Agnostic

The SDK probes `/dev/dri/renderD128` and accepts any VA driver that initialises successfully — including the AMD `radeonsi` driver (`mesa-va-drivers`). There is no Intel-specific filter. On AMD hosts, ensure `LIBVA_DRIVER_NAME=radeonsi` is exported and `vainfo` succeeds before launching the SDK (see [Section 5.2](#52-required-packages--amd)).

### 12.2 GStreamer Decoder Rank Conflict on Ubuntu 22.04

On Ubuntu 22.04, `vaapih264dec` and `avdec_h264` share the same `primary (256)` rank, making `decodebin`'s selection non-deterministic. **Symptom:** VA-API appears enabled but GPU utilisation stays at 0% during decode. **Fix:** set `GST_PLUGIN_FEATURE_RANK=vaapih264dec:257,...` before launching the SDK. Full setup and verification steps are in [Section 7.3](#73-ensuring-vaapih264dec-is-selected-by-decodebin).

### 12.3 OpenCL Platform Conflicts (Multi-GPU / Mixed-Vendor Systems)

If the system has multiple GPUs from different vendors (e.g. a Ryzen APU plus a discrete Radeon, or an Intel CPU with a PCIe Arc card), `clinfo` may list multiple OpenCL platforms. The Voyager SDK selects the OpenCL platform found during compilation. If the wrong platform is selected, the error reported is:

```
what(): No functional OpenCL platform of type '' found.
```

**Workaround options:**

- Use `--disable-opencl` to fall back to CPU processing as a safe baseline.
- Check `/etc/OpenCL/vendors/` to see which ICD files are present. Disable platforms not in use by renaming their `.icd` files.
- On AMD systems with both RustiCL and ROCm installed, ensure only the desired ICD is active.

### 12.4 Kernel Version Requirements

Kernel requirements per platform are listed in [Section 4.1](#41-system-requirements) (Intel) and [Section 5.1](#51-system-requirements) (AMD). Key points: Intel Core Ultra (Meteor Lake) requires kernel ≥ 6.2 — use `linux-generic-hwe-22.04` on Ubuntu 22.04. Lunar Lake requires ≥ 6.12. AMD Ryzen AI 300 (Strix Point) requires ≥ 6.10 with Mesa 25.x.

### 12.5 ROCm Does Not Officially Support AMD APU iGPUs

AMD's official ROCm documentation distinguishes discrete Radeon GPU workloads (officially supported) from Ryzen APU integrated graphics (not officially supported for ROCm). For APU iGPUs, use **Mesa RustiCL** (Option 1 in [Section 5.3](#53-opencl-runtime-options-for-amd-apu)). If ROCm is installed and the iGPU is not enumerated as a ROCm device, configure RustiCL as described in Section 5.3 or use `--disable-opencl` to fall back to CPU processing.

### 12.6 Mesa RustiCL vs. Clover

On Ubuntu 22.04, `mesa-opencl-icd` without `RUSTICL_ENABLE=radeonsi` activates the legacy **Clover** driver (OpenCL 1.1), which lacks extensions required by the Voyager SDK. Always follow the RustiCL setup in [Section 5.3 Option 1](#53-opencl-runtime-options-for-amd-apu) and confirm `clinfo | grep "Platform Name"` reports `rusticl`, not `Clover`.

### 12.7 AMDGPU Power Profile for OpenCL Workloads

By default, the `amdgpu` driver may use a conservative power profile that throttles compute clocks. For sustained OpenCL workloads, switch to the `COMPUTE` power profile:

```bash
# List available profiles
cat /sys/class/drm/card0/device/pp_power_profile_mode

# Enable manual control
echo "manual" | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level

# Apply COMPUTE profile (typically index 5 — verify from the list)
echo "5" | sudo tee /sys/class/drm/card0/device/pp_power_profile_mode
```

To apply persistently at boot, add these commands to a systemd `ExecStart` service or `/etc/rc.local`.

---

## 13. Troubleshooting

### 13.1 Intel-specific Issues

**`/dev/dri/renderD128` does not exist:**

```bash
lsmod | grep i915
sudo modprobe i915
dmesg | grep -i "i915\|drm"
# For Core Ultra: install HWE kernel
sudo apt install -y linux-generic-hwe-22.04 && sudo reboot
```

**`clinfo -l` shows "Number of platforms: 0":**

```bash
cat /etc/OpenCL/vendors/intel.icd
# Expected: /usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so
sudo apt install --reinstall intel-opencl-icd
sudo usermod -a -G render $USER && newgrp render
clinfo -l
```

**`libva error: /dev/dri/renderD128 init failed`:**

```bash
export LIBVA_DRIVER_NAME=iHD   # Gen8+
vainfo
# Or for Gen6/7:
# export LIBVA_DRIVER_NAME=i965 && vainfo
```

### 13.2 AMD-specific Issues

**`/dev/dri/renderD128` does not exist:**

```bash
lsmod | grep amdgpu
dmesg | grep -i "amdgpu\|drm"
# Check for blacklist entries
grep -r amdgpu /etc/modprobe.d/
sudo modprobe amdgpu
```

**`clinfo` shows Clover (OpenCL 1.1):**

```bash
export RUSTICL_ENABLE=radeonsi
clinfo -l
# If still Clover, upgrade Mesa:
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt update && sudo apt install -y mesa-opencl-icd
```

**`clinfo` shows no AMD devices despite amdgpu loaded:**

```bash
groups $USER   # Must include 'render' and 'video'
sudo usermod -a -G render,video $USER && newgrp render

echo $RUSTICL_ENABLE   # Must be: radeonsi

cat /etc/OpenCL/vendors/mesa.icd
# Expected: /usr/lib/x86_64-linux-gnu/libMesaOpenCL.so.1
```

**`vainfo` fails on AMD host (VA-API will not be detected by SDK):**

```bash
sudo apt install -y mesa-va-drivers
export LIBVA_DRIVER_NAME=radeonsi
export LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
vainfo --display drm --device /dev/dri/renderD128
ls /usr/lib/x86_64-linux-gnu/dri/radeonsi_drv_video.so
```

**Voyager SDK: `WARNING: Failed to get OpenCL platforms` on AMD host:**

1. Confirm `RUSTICL_ENABLE=radeonsi` is exported before launching the SDK.
2. Immediate workaround: `--disable-opencl` (CPU fallback).
3. If ROCm is installed alongside RustiCL, check `/etc/OpenCL/vendors/` for conflicting ICD files.

**Strix Point / RDNA 3.5: `gfx1150 not recognised` in LLVM:**

```bash
sudo apt update && sudo apt full-upgrade -y
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt install -y mesa-va-drivers mesa-opencl-icd
glxinfo | grep "Mesa"   # Confirm Mesa 25.x or later
vainfo --display drm --device /dev/dri/renderD128
```

### 13.3 Common Issues (Both Platforms)

**"No functional OpenCL platform of type '' found" (Voyager SDK runtime error):**

1. Run `clinfo -l` to verify at least one platform is visible.
2. Confirm the user is in the `render` group.
3. Use `--disable-opencl` as an immediate workaround.
4. If the SDK was recently upgraded, recompile all models (`rm -rf build/` then `./deploy.py ...`).
5. On AMD hosts: confirm `RUSTICL_ENABLE=radeonsi` is set in the environment where the SDK is launched.

**Disabling OpenCL improves throughput unexpectedly:**

On some systems, failed or inefficient GPU dispatch hurts pipeline performance more than it helps. Adding `--enable-hardware-codec` alongside `--disable-opencl` often recovers throughput by offloading codec work to the hardware decoder:

```bash
./inference.py --disable-opencl --enable-hardware-codec yolov8s-coco media/video.mp4
```

**Multiple OpenCL platforms and wrong one selected:**

See [Section 12.3](#123-opencl-platform-conflicts-multi-gpu--mixed-vendor-systems) for diagnosis and ICD management steps.

---

## 14. Quick Reference

### CLI Flags — Platform Decision Table

| Scenario                                  | Recommended Flags                  |
| ----------------------------------------- | ---------------------------------- |
| Intel host (standard)                     | _(no flags — auto-detect)_         |
| Intel host (force-enable, false negative) | `--enable-vaapi --enable-opencl`   |
| AMD Ryzen APU (VA-API + OpenCL available) | _(no flags — auto-detect)_         |
| AMD Ryzen APU (VA-API unavailable)        | `--disable-vaapi`                  |
| AMD Ryzen APU (OpenCL unavailable)        | `--disable-opencl`                 |
| Any host — CPU only fallback              | `--disable-vaapi --disable-opencl` |
| Troubleshooting OpenCL conflict           | `--disable-opencl`                 |
| Troubleshooting VA-API conflict           | `--disable-vaapi`                  |

### Key Verification Commands — Both Platforms

```bash
# Device nodes
ls -la /dev/dri/

# Kernel driver loaded
lsmod | grep -E "i915|amdgpu"
dmesg | grep -i "i915\|amdgpu\|drm" | tail -20

# User groups
groups $USER   # must contain: render video

# OpenCL
clinfo -l
clinfo | grep -E "Platform Name|Device Name|Device Version"

# VA-API — Intel
vainfo --display drm --device /dev/dri/renderD128

# VA-API — AMD (requires LIBVA_DRIVER_NAME=radeonsi in environment)
LIBVA_DRIVER_NAME=radeonsi vainfo --display drm --device /dev/dri/renderD128

# GStreamer — confirm vaapi plugin is loaded
gst-inspect-1.0 vaapi

# GStreamer — check decoder ranks (Ubuntu 22.04: both may be 256)
gst-inspect-1.0 vaapih264dec | grep -i rank
gst-inspect-1.0 avdec_h264   | grep -i rank

# GStreamer — VA-API smoke test (H.264 FHD decode)
gst-launch-1.0 filesrc location=test_fhd_h264.mp4 ! \
  qtdemux ! h264parse ! vaapih264dec ! vaapipostproc ! fakesink sync=false

# GStreamer — confirm decodebin selects vaapih264dec (requires rank override on 22.04)
GST_DEBUG=decodebin:5 \
gst-launch-1.0 filesrc location=test_fhd_h264.mp4 ! \
  decodebin ! fakesink sync=false 2>&1 | grep -iE "vaapi|avdec|selected|adding"

# GPU utilisation — Intel
sudo intel_gpu_top

# GPU utilisation — AMD
watch -n 1 cat /sys/class/drm/card0/device/gpu_busy_percent
# or: radeontop
```

### Environment Variables Summary

| Variable                  | Intel                                 | AMD                                      |
| ------------------------- | ------------------------------------- | ---------------------------------------- |
| `LIBVA_DRIVER_NAME`       | `iHD` (Gen8+) / `i965` (Gen6–7)       | `radeonsi`                               |
| `LIBVA_DRIVERS_PATH`      | _(not usually required)_              | `/usr/lib/x86_64-linux-gnu/dri`          |
| `RUSTICL_ENABLE`          | _(not applicable)_                    | `radeonsi` (required for RustiCL OpenCL) |
| `OCL_ICD_FILENAMES`       | _(optional — restrict to intel.icd)_  | _(optional — restrict to mesa.icd)_      |
| `GST_PLUGIN_FEATURE_RANK` | `vaapih264dec:257,...` (Ubuntu 22.04) | `vaapih264dec:257,...` (Ubuntu 22.04)    |
