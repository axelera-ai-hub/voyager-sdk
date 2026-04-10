![](/docs/images/Ax_Page_Banner_2500x168_01.png)

# Firmware Update Guide

## Contents
- [Firmware Update Guide](#firmware-update-guide)
  - [Contents](#contents)
  - [Prerequisites (Listed Below)](#prerequisites-listed-below)
  - [Level](#level)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Update Process Overview](#update-process-overview)
  - [Single Metis Device Update](#single-metis-device-update)
    - [Run the Interactive Firmware Update Tool](#run-the-interactive-firmware-update-tool)
    - [Verify the Update](#verify-the-update)
  - [Multiple Metis Devices Update](#multiple-metis-devices-update)
    - [Option 1: Automatic Update (Recommended)](#option-1-automatic-update-recommended)
    - [Option 2: Manual Individual Updates](#option-2-manual-individual-updates)
      - [Step 1: Identify All Device IDs](#step-1-identify-all-device-ids)
      - [Step 2: Update Each Device Individually](#step-2-update-each-device-individually)
      - [Step 3: Verify All Updates](#step-3-verify-all-updates)
  - [Safety and Recovery](#safety-and-recovery)
    - [Stage 1 Update - Metis Firmware Update](#stage-1-update---metis-firmware-update)
    - [Stage 2 Update - Board Controller Firmware Update](#stage-2-update---board-controller-firmware-update)
    - [Safety Recommendations](#safety-recommendations)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Further support](#further-support)

## Prerequisites (Listed Below)
- Voyager SDK installed and virtual environment activated
- Linux system (Windows users must use Linux temporarily)
- Administrative privileges

## Level
**Intermediate** - Comprehensive update procedure with multiple device support

## Overview
> ** WHICH FIRMWARE GUIDE DO YOU NEED?**
> See the [Firmware Update Decision Tree](firmware_update_decision_tree.md) to determine which firmware guide to follow.


> [!TIP]
> **Shorter Guide for routine single-device updates**: See the [Quick Firmware Update Guide](/docs/tutorials/quick_firmware_update.md) for simplified instructions.

> [!NOTE]
> The firmware update procedure is currently only supported on Linux systems. If you are using Windows, you will need to temporarily connect your board to a Linux host to perform the firmware update. After the update is complete, you can reconnect your board to your Windows system. This limitation will be addressed in a future release of the Voyager SDK.

## Prerequisites

To take full advantage of the latest features, improvements, and bug fixes in the current Voyager SDK release, it is strongly recommended to update your board's firmware. Keeping your board's firmware up to date ensures optimal compatibility and performance with the Voyager SDK.

Before you start the firmware update, make sure to:

1. Install the Voyager SDK on your system if you have not done so already. You can follow the steps in the [Voyager SDK install guide](/docs/tutorials/install.md).
2. Activate the Voyager SDK Python virtual environment by running:
```bash
source venv/bin/activate
```

> [!NOTE]
> **All of the following steps must be performed with the Voyager SDK Python virtual environment (`venv`) activated.**
> Ensure you have run `source venv/bin/activate` before starting, and keep the virtual environment active throughout all update steps. If you perform power off and back on, remember to re-activate the environment before continuing.

## Update Process Overview

The interactive flash update script automatically handles all firmware update stages. Simply run the script as instructed below and follow the on-screen instructions - the process is fully guided and will prompt you when power cycling is needed.

For systems with multiple devices, the script can automatically detect and update all connected cards when run without the `--device` option. It will continue checking and updating until all devices are fully updated, requiring you to re-run the script after each power cycle.

## Single Metis Device Update

For systems with a single Metis device connected, follow these steps:

### Run the Interactive Firmware Update Tool

> [!WARNING]
> **Do Not Modify the Flash Update Script**
>
> **Never make local changes** to the `interactive_flash_update` script. Changing the flashing procedure without consulting Axelera can cause system malfunction and may render your board unresponsive. This script contains essential safety checks and proper sequencing that must not be altered under any circumstances.

```bash
axdevice interactive_flash_update
```

The script may request a second power-off and rerun of this script depending on your setup. Please follow the console messages carefully.

### Verify the Update

After the final power off and back on again, verify that the firmware update was successful:

```bash
axdevice
```

You should see output similar to:
```
Device 0: metis-0:1:0 4GiB pcie flver=1.4.0 bcver=7.0 clock=800MHz(0-3:800MHz) mvm=0-3:100%
```

## Multiple Metis Devices Update

For systems with multiple Metis cards or Axelera® AI's PCIe card with 4 Metis® AIPU cores, you have two options: automatic update of all devices or individual device updates.

### Option 1: Automatic Update (Recommended)

Run the script without specifying a device to automatically detect and update all connected cards:

```bash
axdevice interactive_flash_update
```

The script will:
- Automatically detect all connected Metis devices
- Update firmware for all devices that need updates
- Prompt you to power cycle when required
- Continue updating all devices until everything is fully updated

> [!IMPORTANT]
> **After each power cycle, you must re-run the script** to continue the update process. The script will check all devices and update any that still need updates. Keep running the script after each power cycle until it reports "ALL DEVICES ARE FULLY UPDATED!"

### Option 2: Manual Individual Updates

If you prefer to update devices individually, follow these steps:

#### Step 1: Identify All Device IDs

First, identify the device IDs of all Metis® AIPU cores:

```bash
axdevice
```

This command will list all detected devices with their IDs:
```
Device 0: metis-0:6c:0 board_type=pcie fwver='1.3.2' clock=800MHz(0-3:800MHz) mvm=0-3:100%
Device 1: metis-0:6d:0 board_type=pcie fwver='1.3.2' clock=800MHz(0-3:800MHz) mvm=0-3:100%
Device 2: metis-0:6e:0 board_type=pcie fwver='1.3.2' clock=800MHz(0-3:800MHz) mvm=0-3:100%
Device 3: metis-0:6f:0 board_type=pcie fwver='1.3.2' clock=800MHz(0-3:800MHz) mvm=0-3:100%
```

#### Step 2: Update Each Device Individually

Update the firmware for each device ID individually using the `--device` option:

```bash
axdevice interactive_flash_update --device <device-id>
```

For example, using the device IDs from step 1:

```bash
axdevice interactive_flash_update --device metis-0:6c:0
```
Go through all rounds. Then continue with each one by one:
```bash
axdevice interactive_flash_update --device metis-0:6d:0
```
```bash
axdevice interactive_flash_update --device metis-0:6e:0
```
```bash
axdevice interactive_flash_update --device metis-0:6f:0
```

> [!IMPORTANT]
> You must execute the firmware update command for each of the 4 device IDs to ensure all Metis® AIPU cores on the PCIe card are properly updated.

#### Step 3: Verify All Updates

After updating all devices, verify that all firmware updates were successful:

```bash
axdevice
```

## Safety and Recovery

> [!WARNING]
> **Never interrupt power during firmware updates**

Firmware updates carry significant risks. Understanding the safety mechanisms and recovery procedures is crucial:

### Stage 1 Update - Metis Firmware Update

- **Failsafe Mechanism**: The update uses dual-boot regions in flash memory and updates the entire safety mechanism
- **Recovery**: If power is lost during update, the previous firmware version will be still available and can be used to retry the update process
- **Bricking Risk**: **LOW** 

### Stage 2 Update - Board Controller Firmware Update

- **Failsafe Mechanism**: No - Single region update
- **Recovery**: There is no mechanism for recovery. Contact Axelera AI for assistance
- **Bricking Risk**: **HIGH** - Ensure uninterrupted power supply throughout the process

### Safety Recommendations

1. **Use a UPS (Uninterruptible Power Supply)** to protect against power interruptions
2. **Avoid system updates or reboots** during the firmware update process
3. **Close unnecessary applications** to reduce system load
4. **Monitor system temperature** to prevent thermal shutdowns
5. **Follow all prompts carefully** and do not interrupt the process

## Troubleshooting

### Common Issues

If you encounter issues during the firmware update process:

1. **Check System Requirements**: Ensure your system meets all prerequisites
2. **Verify Environment**: Confirm the Voyager SDK environment is properly activated
3. **Review Logs**: Check for any error messages in the terminal output


## Next Steps
- **After successful update**: Verify with `axdevice` command
- **For future updates**: Use [Quick Firmware Update Guide](quick_firmware_update.md) if single device
- **Continue development**: Return to [Quick Start Guide](quick_start_guide.md)
- **If issues persist**: Check Troubleshooting section or contact support

## Related Documentation
**Firmware Guides:**
- **[Firmware Update Decision Tree](firmware_update_decision_tree.md)** - START HERE to choose the right guide
- [Quick Firmware Update](quick_firmware_update.md) - Simpler alternative for single-device routine updates

**Tutorials:**
- [Installation Guide](install.md) - SDK must be installed
- [Quick Start Guide](quick_start_guide.md) - Verify functionality after update

**References:**
- [AxDevice API](../reference/axdevice.md) - Verify firmware version after update

## Further support

For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).

For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).
