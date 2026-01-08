![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Enable Card Firmware Update

## Contents
- [Enable Card Firmware Update](#enable-card-firmware-update)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Level](#level)
  - [Overview](#overview)
  - [Steps to Enable Firmware Updates](#steps-to-enable-firmware-updates)
    - [Step 1: Install Voyager SDK](#step-1-install-voyager-sdk)
    - [Step 2: Download the Bootloader Update File](#step-2-download-the-bootloader-update-file)
    - [Step 3: Enable Bootloader Updates](#step-3-enable-bootloader-updates)
    - [Step 4: Power Cycle Your System](#step-4-power-cycle-your-system)
  - [Verification](#verification)
  - [Troubleshooting](#troubleshooting)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Further support](#further-support)

> ** WHICH FIRMWARE GUIDE DO YOU NEED?**  
> See the [Firmware Update Decision Tree](firmware_update_decision_tree.md) to determine which firmware guide to follow.

## Prerequisites

- Administrative privileges on your system
- Internet connection to download required files
- Voyager SDK will be installed as part of this guide

NOTE: this guide is meant for single card systems. If you use multiple cards, please connect them one-by-one and run the `enable_bootloader_update.sh` script for each.

## Level
**Beginner** - One-time setup, follow step-by-step instructions

## Overview
Some customers may have boards where updating the flash is not enabled by default. This guide will help you enable firmware updates on these boards.


## Steps to Enable Firmware Updates

### Step 1: Install Voyager SDK

First, you need to install the Voyager SDK following the official setup instructions:

1. Visit the [Voyager SDK Quick Start Guide](/docs/tutorials/quick_start_guide.md#setup)
2. Follow the setup instructions to install the Voyager SDK on your system
3. Make sure to run the Step 2 and Step 3 below in a prompt with the (venv) activated from the guide above

### Step 2: Download the Bootloader Update File

Download the required bootloader update file:

```bash
wget https://axelera-public.s3.eu-central-1.amazonaws.com/aipu_firmware_enabler/voyager-sdk-v1.4.0/enable_bootloader_update.sh
chmod +x enable_bootloader_update.sh
```

### Step 3: Enable Bootloader Updates

Use the script previously downloaded to enable bootloader updates:

```bash
./enable_bootloader_update.sh
```

This command will enable the bootloader update functionality on your board.

### Step 4: Power Cycle Your System

After running the bootloader update command:

1. **Power off** your PC completely
2. **Power on** your PC
3. Wait for the system to fully boot up

## Verification

After completing these steps, you should be able to use the standard firmware flash update procedures. Refer to the [Firmware Flash Update Guide](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.4/docs/tutorials/firmware_flash_update.md) for detailed instructions on updating your firmware.

## Troubleshooting

If you encounter any issues:

1. Ensure the Voyager SDK is properly installed
2. Verify that the bootloader update file was downloaded successfully
3. Check there are no errors displayed on screen
4. Make sure to perform a complete power off and back on again (not just a restart)

## Next Steps
- **After enabling updates**: Use [Quick Firmware Update Guide](quick_firmware_update.md) for future updates
- **For complex scenarios**: Use [Full Firmware Update Guide](firmware_flash_update.md)
- **Verify hardware**: Run `axdevice` to confirm board is detected
- **Start using SDK**: [Quick Start Guide](quick_start_guide.md)

## Related Documentation
**Firmware Guides:**
- **[Firmware Update Decision Tree](firmware_update_decision_tree.md)** - Start here to choose the right guide
- [Quick Firmware Update](quick_firmware_update.md) - For boards already enabled (after completing this guide)
- [Firmware Update Guide (Full)](firmware_flash_update.md) - Comprehensive update procedure

**Tutorials:**
- [Installation Guide](install.md) - Step 1 of this guide installs the SDK
- [Quick Start Guide](quick_start_guide.md) - Verify installation after enabling updates

**References:**
- [AxDevice API](../reference/axdevice.md) - Verify hardware detection

## Further support

For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).

For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).
