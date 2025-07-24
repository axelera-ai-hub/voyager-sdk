![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Board firmware update procedure
> [!WARNING]
> **Before attempting any firmware updates, you must ensure that your board is enabled for updates.**
> 
> If you wish to enable firmware updates on your board, please carefully follow the steps in the [Enable Card Firmware Update Guide](/docs/tutorials/enable_updates.md) **before proceeding with any update attempts.**
> 
> Attempting to update the flash on boards without first enabling updates may result in a bricked board and render your hardware unusable. Always verify that update enablement has been completed successfully prior to flashing firmware. This procedure only needs to be done once for each board.

Some older Axelera AI development boards require a firmware update to be used with later versions of the
Voyager SDK. If required, follow these steps:

1. [Install the Voyager SDK](/docs/tutorials/install.md) (if not already installed on your system)

2. Activate the Voyager SDK development environment.

```
source venv/bin/activate
```

3. Download the board firmware to your development system.

```
wget https://axelera-public.s3.eu-central-1.amazonaws.com/aipu_firmware/voyager-sdk-v1.2.0/firmware_release_public_v1.2.0.tar.gz
```

4. Extract the board firmware to the current directory.

```
tar xzvf firmware_release_public_v1.2.0.tar.gz
```

5. Run the firmware update tool to flash the firmware to your board.

```
cd firmware_release_public_v1.2.0
./flash_update.sh flash_bundle.img
```

The firmware flashing tool takes up to two minutes to run and on success outputs the message `flash success`.
