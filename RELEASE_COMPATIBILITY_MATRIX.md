# Release Compatibility Matrix

The following compatibility matrix describes the recommended and supported versions of firmware and driver per Voyager SDK release. Consult it before installing or upgrading a card or Voyager SDK release.

> **Tip!** `axversion` outputs the SDK version and `axversion --driver` outputs the driver version. `axdevice` outputs the firmware and board controller firmware version.

Recommended = version shipped with the SDK release.

Supported = versions tested with the SDK release.

Note: Other versions may work but are not actively tested. An upgrade of the card's flashed firmware and board controller firmware to a compatible version using `axdevice interactive_flash_update` script is advised.

| Release | Board controller (Recommended) | Board controller (Supported) | Flashed Firmware (Recommended) | Flashed Firmware (Supported) | PCIe driver - Linux (Recommended) | PCIe driver - Linux (Supported) | PCIe driver - Windows (Recommended) | PCIe driver - Windows (Supported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1.6.0 | 7.4 | 7.0 | 1.6.0 | 1.5.0<br>1.4.0 | 1.4.16 | 1.4.10<br>1.4.4 | 1.3.4 | 1.3.1<br>1.3.0 |
| v1.6.1 | 7.4 | 7.0 | 1.6.0 | 1.5.0<br>1.4.0 | 1.4.17 | 1.4.16<br>1.4.10<br>1.4.4 | 1.3.11 | 1.3.4<br>1.3.1<br>1.3.0 |
| v1.7.0 | 7.4 | 7.0 | 1.7.0 | 1.6.0<br>1.5.0<br>1.4.0 | 1.5.7 | 1.5.5 | 1.3.11 | 1.3.5<br>1.3.4<br>1.3.1<br>1.3.0 |
| v1.8.0 | 7.4 | 7.0 | 1.8.0 | 1.7.0<br>1.6.0<br>1.5.0<br>1.4.0 | 1.6.2 | 1.6.2 | 1.3.14 | 1.3.14 |

Notes on how to read this table:

- All values are MINIMUMS: recommended = version we recommend users run; supported = lowest version that still works (floor).
- Source: each component's `compatibility.yaml`, read at the exact commit pinned in `software-platform` `west.yml` for that release.
- Board controller: supported floor held at 7.0 across every release.
- Firmware column = device firmware bootloader/stage0 (the only firmware version the firmware `compatibility.yaml` declares).
- Driver: `compatibility.yaml` `version` field = recommended; `supported` = minimum. Windows recommended plateaus at 1.3.11 from v1.6.1.
- Recommended version is the version that the SDK has been shipped with.

## v1.8.0 driver requirement

For both Linux and Windows, the minimum *supported* driver version in v1.8.0 is equal to the shipped version, so **no earlier driver is accepted**:

- Linux: install `metis-dkms` **1.6.2**
- Windows: install **MetisDriver-1.3.14**

Board controller firmware requires **no** update in this release (the recommended version is still 7.4).

See the [Voyager SDK release notes](RELEASE_NOTES.md) for the full release detail.
