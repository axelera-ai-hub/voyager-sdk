![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Board firmware update procedure

Some older Axelera AI development boards require a firmware update to be used with later versions of the
Voyager SDK. If required, follow these steps:

1. [Install the Voyager SDK](/docs/tutorials/install.md) (if not already installed on your system)

2. Activate the Voyager SDK development environment.

```
source venv/bin/activate
```

3. Download the board firmware to your development system.

```
wget https://axelera-public.s3.eu-central-1.amazonaws.com/built_metis_firmware/voyager-sdk-v1.2.0/firmware_release_public_v1.2.0.tar.gz
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
