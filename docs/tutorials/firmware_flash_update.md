![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Board firmware update procedure
> [!WARNING]
> **Before attempting any firmware updates, you must ensure that your board is enabled for updates.**
> 
> If you wish to enable firmware updates on your board, please carefully follow the steps in the [Enable Card Firmware Update Guide](/docs/tutorials/enable_updates.md) **before proceeding with any update attempts.**
> 
> Attempting to update the flash on boards without first enabling updates may result in a bricked board and render your hardware unusable. Always verify that update enablement has been completed successfully prior to flashing firmware. This procedure only needs to be done once for each board.

> [!NOTE]  
> The firmware update procedure is currently only supported on Linux systems. If you are using Windows, you will need to temporarily connect your board to a Linux host to perform the firmware update. After the update is complete, you can reconnect your board to your Windows system. This limitation will be addressed in a future release of the Voyager SDK.


Some older Axelera AI development boards require a firmware update to be used with later versions of the
Voyager SDK. If required, follow these steps:

1. [Install the Voyager SDK](/docs/tutorials/install.md) (if not already installed on your system)

2. Activate the Voyager SDK development environment.

```
source venv/bin/activate
```

3. Download the board firmware to your development system.

```
wget https://axelera-public.s3.eu-central-1.amazonaws.com/aipu_firmware/voyager-sdk-v1.3.2/firmware_release_public_v1.3.2.tar.gz
```

4. Extract the board firmware to the current directory.

```
tar xzvf firmware_release_public_v1.3.2.tar.gz
```
> [!NOTE]  
> After extracting the firmware, you need to make the flash update script executable before proceeding to the next step:
> 
> ```bash
> chmod +x firmware_release_public_v1.3.2/flash_update.sh
> ```

### Flash the firmware

5. Run the firmware update tool to flash the firmware to your board. If you have multiple Metis devices connected, you can specify the device by using the `--device` option.

```
cd firmware_release_public_v1.3.2
chmod u+x flash_update.sh
./flash_update.sh --fw-update
```

The firmware flashing tool takes up to two minutes to run and on success outputs the message `flash success`.

### Flash the board controller firmware

> This step will only work if the pre-installed board controller firmware is at least `v1.0`. Otherwise contact an FAE or consult the community support.


6. To figure out if the board will support an update, run the following command
```
axdevice
Device 0: metis-0:1:0 4GiB pcie flver=1.3.2 bcver=1.0 clock=800MHz(0-3:800MHz) mvm=0-3:100%

``` 
which should report the `bcver` of at least `1.0`.

7. Run the firmware update tool to flash the board controller firmware. If you have multiple Metis devices connected, you can specify the device by using the `--device` option.
```
cd firmware_release_public_v1.3.2
chmod u+x flash_update.sh
./flash_update.sh --bc-update
```

> NOTICE: This is not a failsafe or revertible function. Removing power during this process might brick your device. Verify to proceed with "y" when prompted. If you run into any issues, contact an FAE or consult the community support.

After this step, the board will reboot. Next, check if the board controller version was successfully updated and the board is responsive.
```
axdevice
Device 0: metis-0:1:0 4GiB pcie flver=1.3.2 bcver=1.4 clock=800MHz(0-3:800MHz) mvm=0-3:100%
```
