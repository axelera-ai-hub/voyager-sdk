![](/docs/images/Ax_Page_Banner_2500x168_01.png)

# Firmware Update Decision Tree

## Contents
- [Firmware Update Decision Tree](#firmware-update-decision-tree)
  - [Contents](#contents)
  - [Prerequisites for All Firmware Updates](#prerequisites-for-all-firmware-updates)
  - [Overview](#overview)
  - [Decision Flowchart](#decision-flowchart)
  - [Quick Reference Table](#quick-reference-table)
  - [Two-Step Process](#two-step-process)
    - [Step 1: Update Firmware (Simple Cases)](#step-1-update-firmware-simple-cases)
    - [Step 2: Update Firmware (Complex Cases)](#step-2-update-firmware-complex-cases)
  - [Common Scenarios](#common-scenarios)
    - [Scenario A: Single Device, Routine Update](#scenario-a-single-device-routine-update)
    - [Scenario B: Multiple Devices](#scenario-b-multiple-devices)
    - [Scenario C: Windows User](#scenario-c-windows-user)
    - [Scenario D: Board Not Working / Recovery](#scenario-d-board-not-working--recovery)
  - [Key Safety Rules](#key-safety-rules)
  - [Still Unsure?](#still-unsure)
  - [Document Links](#document-links)

## Prerequisites for All Firmware Updates

Before starting any firmware procedure:
-  Voyager SDK installed
-  Virtual environment activated (`source venv/bin/activate`)
-  Administrative privileges on your system
-  Stable power supply (no risk of power loss during update)
-  Linux system (or Linux host for Windows users)

---

## Overview
This guide helps you determine which firmware update procedure to follow based on your board's current state.


## Decision Flowchart

```
START: Do you need to update firmware?
│
├─> SIMPLE UPDATE (single device, routine update)
│   │
│   └─> Go to: Quick Firmware Update Guide
│       (docs/tutorials/quick_firmware_update.md)
│
└─> COMPLEX SCENARIO (multiple devices)
    │
    └─> Go to: Full Firmware Update Guide
        (docs/tutorials/firmware_flash_update.md)
```

---

## Quick Reference Table

| Your Situation | Which Guide | Why |
|---------------|-------------|-----|
| **Routine update, single device** | [Quick Update](quick_firmware_update.md) | Fastest path for simple updates |
| **Multiple devices** | [Full Update](firmware_flash_update.md) | Handles complex multi-device scenarios |
| **First time updating this board** | [Full Update](firmware_flash_update.md) | Comprehensive instructions with safety checks |
| **Recovery needed** | [Full Update](firmware_flash_update.md) | Includes troubleshooting and recovery steps |
| **Windows user** | [Full Update](firmware_flash_update.md) | Contains Linux requirement note and workarounds |

---

## Two-Step Process

### Step 1: Update Firmware (Simple Cases)

**Document:** [Quick Firmware Update Guide](quick_firmware_update.md)

**When to use:**
- Single device system
- Routine update to a newer firmware version

**What it does:**
- Runs `axdevice interactive_flash_update`
- Guides you through power cycling
- Updates to latest firmware version

---

### Step 2: Update Firmware (Complex Cases)

**Document:** [Firmware Update Guide](firmware_flash_update.md)

**When to use:**
- Multiple Metis cards or PCIe card with 4 cores
- Need troubleshooting or recovery procedures
- Linux requirement matters (Windows users)

**What it does:**
- Comprehensive update procedure
- Per-device targeting with `--device` flag
- Automatic multi-device detection
- Safety checks and recovery procedures
- Troubleshooting section

---

## Common Scenarios

### Scenario A: Single Device, Routine Update
1. Follow [Quick Update Guide](quick_firmware_update.md)

### Scenario B: Multiple Devices
1. Follow [Full Update Guide](firmware_flash_update.md) - use automatic multi-device update or per-device targeting

### Scenario C: Windows User
1. Must use Linux system temporarily for firmware update
2. Follow [Full Update Guide](firmware_flash_update.md) on Linux
3. Reconnect board to Windows after update complete

### Scenario D: Board Not Working / Recovery
1. Follow [Full Update Guide](firmware_flash_update.md)
2. Check Troubleshooting section
3. Contact support if recovery fails

---

## Key Safety Rules

 **Never modify:** Do not edit `interactive_flash_update.sh` - contact Axelera support instead

 **Always back up:** Save important work before firmware updates

 **Power cycling:** Follow power cycle instructions exactly - complete power off required (not restart)

---

## Still Unsure?

**Default recommendation:** If you're uncertain, use the [Full Update Guide](firmware_flash_update.md) - it covers all cases including troubleshooting.

**Contact support:** If you encounter any issues or have questions about your specific setup, contact Axelera AI support before proceeding.

---

## Document Links

- [Quick Firmware Update Guide](/docs/tutorials/quick_firmware_update.md)
- [Firmware Update Guide (Full)](/docs/tutorials/firmware_flash_update.md)
- [Installation Guide](/docs/tutorials/install.md)
- [Quick Start Guide](/docs/tutorials/quick_start_guide.md)
