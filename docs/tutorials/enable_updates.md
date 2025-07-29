![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Enable Card Firmware Update

Some customers may have boards where updating the flash is not enabled by default. This guide will help you enable firmware updates on these boards.

## Prerequisites

Before proceeding, ensure you have:
- Administrative privileges on your system
- Internet connection to download required files

## Steps to Enable Firmware Updates

### Step 1: Install Voyager SDK

First, you need to install the Voyager SDK following the official setup instructions:

1. Visit the [Voyager SDK Quick Start Guide](/docs/tutorials/quick_start_guide.md#setup)
2. Follow the setup instructions to install the Voyager SDK on your system
3. Make sure to run the Step 2 and Step 3 below in a prompt with the (venv) activated from the guide above

### Step 2: Download the Bootloader Update File

Download the required bootloader update file using curl:

```bash
wget https://axelera-public.s3.eu-central-1.amazonaws.com/aipu_firmware_enabler/voyager-sdk-v1.2.0/enable_bootloader_update.sh
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

After completing these steps, you should be able to use the standard firmware flash update procedures. Refer to the [Firmware Flash Update Guide](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.3/docs/tutorials/firmware_flash_update.md) for detailed instructions on updating your firmware.

## Troubleshooting

If you encounter any issues:

1. Ensure the Voyager SDK is properly installed
2. Verify that the bootloader update file was downloaded successfully
3. Check that the `triton_multi_ctx` command completed without errors
4. Make sure to perform a complete power cycle (not just a restart)

## Support

If you continue to experience issues after following these steps, please contact Axelera AI support for assistance. 
