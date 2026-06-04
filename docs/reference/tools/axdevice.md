---
title: "axdevice"
---
# axdevice — Device Detection and Configuration

The `axdevice` tool lists and configures all Axelera [AIPU](../../glossary.md#aipu) devices installed in your system. It is the first tool you should run after installation to confirm your hardware is detected.

## Quick reference

```bash
# List all devices
axdevice

# Target a specific device
axdevice -d0
axdevice -dmetis-0:3:0

# Generate a support report
axdevice --report
```

## What the output means

```
Device 0: metis-0:1:0 4GiB pcie flver=1.3.0 bcver=1.4 clock=800MHz(0-3:800MHz) mvm=0-3:100%
```

| Field | Meaning |
|-------|---------|
| `Device 0` | Device index — use this with `-d0` to target it |
| `metis-0:1:0` | Device name — PCIe bus address (bus:device:function) |
| `4GiB` | On-device DDR memory |
| `pcie` | Connection type (PCIe or M.2) |
| `flver=1.3.0` | Firmware loader version |
| `bcver=1.4` | Board controller version |
| `clock=800MHz` | Current clock speed |
| `(0-3:800MHz)` | Per-core clock speeds (cores 0 through 3) |
| `mvm=0-3:100%` | MVM (Matrix-Vector Multiply) utilization per core |

## Options

### Device selection

```bash
axdevice -d <device>
```

Select a specific device by index or name. If omitted, commands apply to all devices.

```bash
axdevice -d1 --set-clock 400        # By index
axdevice -dmetis-0:3:0 --set-clock 400  # By name
```

### Clock speed

```bash
axdevice --set-clock <MHz>
```

Set the clock speed for all cores. Valid values: **100, 200, 400, 600, 700, 800**.

```bash
axdevice --set-core-clock <core-spec>
```

Set clock speed per core:

```bash
axdevice --set-core-clock 0-1:800,2-3:400   # Cores 0-1 at 800, cores 2-3 at 400
axdevice --set-core-clock 600                # All cores at 600
```

### MVM limitation

```bash
axdevice --set-mvm-limitation <core-spec>
```

Restrict MVM utilization per core (1-100%):

```bash
axdevice --set-mvm-limitation 0:100,1-3:50   # Core 0 full, cores 1-3 at 50%
axdevice --set-mvm-limitation 75              # All cores at 75%
```

> [!IMPORTANT]
> **Why limit MVM?**
> Reducing MVM utilization lowers power consumption and heat. Useful when running multiple models or in thermally constrained environments.


### Device recovery

| Command | What it does |
|---------|-------------|
| `--pcie-scan` | List all Axelera devices on the PCIe bus (uses `lspci`) |
| `--reload-firmware` | Reload device runtime firmware |
| `--pcie-rescan` | Remove and re-enumerate all Axelera PCIe devices |
| `--refresh` | Run `--pcie-rescan` then `--reload-firmware` (try up to 3 times) |
| `--reboot` | Power cycle the device. Interrupts any in-flight inference. |

> [!TIP]
> If `axdevice` shows no devices after boot, run `axdevice --refresh` up to three times. Some hosts have PCIe enumeration timing issues on cold boot.


> [!WARNING]
> **Metis PCIe 4-AIPU cards**
> Some early Metis PCIe 4-AIPU cards may fail to recover when cascaded with other PCIe devices on the same bus. In this case, disable the multi-device service to prevent interference with the recovery process:
>
> ```bash
> sudo systemctl disable axelera-multi-device
> ```


### Support report

```bash
axdevice --report
```

Generates a ZIP file (e.g., `report-2025-06-05_11_56_12.zip`) containing:
- System information (OS, RAM, kernel version)
- `lspci` output
- Firmware versions and device configuration
- Device error logs
- Kernel `dmesg` log

Attach this file when contacting [Axelera Support](https://support.axelera.ai).

## Important note

When using [inference.py](inference-py.md) with a YAML [pipeline](../../glossary.md#pipeline), the pipeline automatically configures the device at startup. This overwrites any clock or MVM settings you set manually. To prevent this, set:

```bash
export AXELERA_CONFIGURE_BOARD=0
```

## See also

- [Verify Setup](../../getting-started/verify-setup.md) — uses `axdevice` as a verification step
- [Glossary: AIPU](../../glossary.md#aipu) — what the AI Processing Unit is
