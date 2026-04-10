![](/docs/images/Ax_Page_Banner_2500x168_01.png)

# Quick Firmware Update Guide

## Contents
- [Quick Firmware Update Guide](#quick-firmware-update-guide)
  - [Contents](#contents)
  - [Prerequisites (Listed Below)](#prerequisites-listed-below)
  - [Level](#level)
  - [Overview](#overview)
  - [Quick Update Steps](#quick-update-steps)
    - [1. Activate the Voyager SDK Environment](#1-activate-the-voyager-sdk-environment)
    - [2. Run the Interactive Firmware Update Tool](#2-run-the-interactive-firmware-update-tool)
    - [3. Verify the Update](#3-verify-the-update)
  - [Important Notes](#important-notes)
    - [Multi-Card Systems](#multi-card-systems)
    - [Safety Reminder](#safety-reminder)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Getting Help](#getting-help)

## Prerequisites (Listed Below)
- Voyager SDK is installed on your system
- You have the Voyager SDK Python virtual environment available

## Level
**Beginner** - Simple update procedure for single-device systems

## Overview

> ** WHICH FIRMWARE GUIDE DO YOU NEED?**
> See the [Firmware Update Decision Tree](firmware_update_decision_tree.md) to determine which firmware guide to follow.


## Quick Update Steps

### 1. Activate the Voyager SDK Environment

```bash
source venv/bin/activate
```

### 2. Run the Interactive Firmware Update Tool

```bash
axdevice interactive_flash_update
```

Follow the on-screen instructions. The script will guide you through the entire process, including when power cycling is needed.

### 3. Verify the Update

After the final power off and back on, verify that the firmware update was successful:

```bash
axdevice
```

You should see output showing the updated firmware version.

## Important Notes

### Multi-Card Systems

If you have a system with **multiple Metis cards** or an **Axelera® AI PCIe card with 4 Metis® AIPU cores**, you **MUST** still follow the detailed steps in the [Multiple Metis Devices Update](/docs/tutorials/firmware_flash_update.md#multiple-metis-devices-update) section of the main firmware update guide.

Each Metis core must be updated individually using the `--device` option.

### Safety Reminder

- Never interrupt power during firmware updates
- Use a UPS if possible to prevent power interruptions
- Follow all prompts carefully and do not interrupt the process

## Next Steps
- **After update**: Verify with `axdevice` command
- **If issues occur**: Use [Full Firmware Update Guide](firmware_flash_update.md) for troubleshooting
- **Continue development**: Return to [Quick Start Guide](quick_start_guide.md)

## Related Documentation
**Firmware Guides:**
- **[Firmware Update Decision Tree](firmware_update_decision_tree.md)** - START HERE to choose the right guide
- [Firmware Update Guide (Full)](firmware_flash_update.md) - For multi-device systems or recovery

**Tutorials:**
- [Installation Guide](install.md) - SDK must be installed
- [Quick Start Guide](quick_start_guide.md) - Verify functionality after update

**References:**
- [AxDevice API](../reference/axdevice.md) - Verify firmware version after update

## Getting Help

- **Community Support**: Visit the [Axelera AI Community](https://community.axelera.ai/)
- **Technical Support**: Contact your FAE or Axelera AI support team
- **Complete Guide**: For detailed information, see the [full firmware update guide](/docs/tutorials/firmware_flash_update.md)
