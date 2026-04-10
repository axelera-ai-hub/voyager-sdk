![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# AxDevice

- [AxDevice](#axdevice)
  - [Configuration \& Advanced Usage](#configuration--advanced-usage)
    - [--device `<device>`, -d `<device>`](#--device-device--d-device)
    - [--set-clock `<clock>`](#--set-clock-clock)
    - [--set-core-clock `<core-clock>`](#--set-core-clock-core-clock)
    - [--set-mvm-limitation `<core-mvm>`](#--set-mvm-limitation-core-mvm)
    - [--pcie-scan](#--pcie-scan)
    - [--reload-firmware](#--reload-firmware)
    - [--pcie-rescan](#--pcie-rescan)
    - [--refresh](#--refresh)
    - [--reboot](#--reboot)
    - [--report](#--report)

The `axdevice` tool is used to enumerate and configure Axelera devices. In the basic case, it
is used to list all the available Axelera devices currently installed in the system.

Note that when using the YAML-based Pipeline Builder the pipeline at initialization will configure
the device to ensure it is configured correctly. The settings used are configured in the
pipeline YAML. However, this means that changes to core clock, and MVM limitation
will be overwritten. This device configuration can be prevented by setting the environment
variable `AXELERA_CONFIGURE_DEVICE=0`. `axrunmodel` does not configure the device so this
is not necessary.

Basic usage of the `axdevice` tool is as follows:

```bash
axdevice
```

As no arguments were provided, all devices found will be listed. Information about each device, such
as its name, amount of DDR, type, firmware versions, clock speed and the MVM limitation will be provided.

This can be used to quickly determine which devices are available, along with their names and ID,
which is useful if you wish to run subsequent commands on a specific device. For example:

```bash
./inference.py -dmetis-0:3:0 ...
```

## Configuration & Advanced Usage

The following arguments can be used to configure devices, enumerate them, or refresh device state.
Configurations can be made on a per device basis, or by default applied to all available
devices. A full list of arguments can also be obtained from `axdevice --help`:

### --device `<device>`, -d `<device>`

The 'device' argument is used to select a device to configure. The device
can be referred to by its index or name, obtained by running `axdevice` without arguments.

For example:

```bash
$ axdevice
Device 0: metis-0:1:0 4GiB pcie flver=1.3.0 bcver=1.4 clock=800MHz(0-3:800MHz) mvm=0-3:100%
Device 1: metis-0:3:0 4GiB pcie flver=1.3.0 bcver=1.4 clock=800MHz(0-3:800MHz) mvm=0-3:100%
$ axdevice -d1 --set-clock 400
$ axdevice -dmetis-0:3:0 --set-clock 400
```

If the 'device' argument is omitted, the configuration will be applied to all the currently
enumerated devices

### --set-clock `<clock>`

The 'set clock' argument is used to set the clock speed of the device(s). It is given as a
frequency in MHz.

The clock speed can be set to 100, 200, 400, 600, 700 or 800.

### --set-core-clock `<core-clock>`

The 'set core clock' argument is used to set the clock speed of a specific cores or all
cores in the device. Clock speeds should be given in MHz.

The value can be a list of core indexes and clock speeds, or just a clock speed. For example:

```bash
axdevice --set-core-clock 0-1:800,2-3:400
axdevice --set-core-clock 600
```

The first command will configure cores 0 and 1 to 800MHz, and cores 2 and 3 to 400MHz.

The second command will configure all cores to 600Mhz.

### --set-mvm-limitation `<core-mvm>`

The 'set MVM limitation' argument is used to restrict the MVM utilization of specific cores
or all cores in the device. The limit should be given as a percentage (%) in the range [1, 100].

The value can be a list of core indexes and MVM limitations, or just an MVM limitation.
For example:

```bash
axdevice --set-mvm-limitation 0:100,1-3:50
axdevice --set-mvm-limitation 75
```

The first command will leave core 0 unrestricted in MVM utilization, but restrict cores
1, 2 and 3 to 50% utilization.

The second command will restrict all cores to 75% utilization of the MVM.

### --pcie-scan

The 'PCIE scan' argument uses `lspci` to get a list of devices found on the PCIE bus, and lists
all Axelera PCIE/M.2 devices which can be found.

Note: `axdevice` without arguments does not use `lspci` but directly enumerates the PCIE
devices for itself. Normally these lists will match. In the case of driver, permission,
or other issues, some devices may be listed with `--pcie-rescan`, but not by `axdevice`
without arguments.

### --reload-firmware

The 'reload firmware' argument is used to reload the device runtime firmware of the enumerated
Axelera PCIE/M.2 device(s).

It can be used on its own to reload all devices, or with the `--device` argument to only
reload the device runtime firmware of a specific device.

The devices will be listed after running.

This command is rarely required.  To force a rescan of available devices it is generally better
to use `--refresh`.

### --pcie-rescan

The 'PCIE rescan' argument can be used to remove all the enumerated Axelera PCIE/M.2 devices,
and perform a rescan to discover and enumerate them again.

The re-enumerated devices will be listed after running.

### --refresh

The 'refresh' argument will run the PCIE rescan argument followed by the reload firmware argument.
This may be useful if you are experiencing issues with the device(s). In particular on some hosts
there are issues with PCIE enumeration after a boot that means it can take up to 3 calls of `--refresh`
to get devices enumerated properly.

_* Some early __4-Metis PCIe boards__ may fail to recover when cascaded with other PCIe devices on the same bus.
In this case, disable the axelera-multi-device service to prevent interference with the recovery process:
`sudo systemctl disable axelera-multi-device`_

The re-enumerated devices will be listed after running.

### --reboot

The 'reboot' argument will force the Axelera device(s) to power cycle.  This is sometimes necessary
to fix connection issues if `--refresh` does not recover the device.

The re-enumerated devices will be listed after running.

### --report

This command will create a zip file in the current directory, for example `report-2025-06-05_11_56_12.zip`.

This file includes two files :

- kernel-log-2025-06-05_11_56_12.txt - the dmesg log
- report-2025-06-05_11_56_12.txt - host and Axelera information
  - Host OS, RAM, version etc
  - The current environment
  - Axelera package information
  - The result of calling `lspci -tv`
  - Axelera firmware versions and configuration
  - Axelera device error logs

We suggest adding this zip file when you report an issue to Axelera Customer Support.
