# Voyager SDK Release Compatibility Matrix

This page describes the recommended and supported versions of firmware and driver per Voyager SDK release for Axelera Metis AI Accelerator Cards. Consult it before installing or upgrading a card or a Voyager SDK release.

It is maintained cumulatively across releases and is updated with each new Voyager SDK version. It is not specific to any single release.

## Reading this matrix

- **Recommended** = the version shipped with the SDK release.
- **Supported** = the versions tested with the SDK release.

Other versions may work but are not actively tested. An upgrade of the card's flashed firmware and board controller firmware to a compatible version using the `axdevice interactive_flash_update` script is advised.

> **Tip!** `axversion` outputs the SDK version and `axversion --driver` outputs the driver version. `axdevice` outputs the firmware and board controller firmware version.

## Metis compatibility matrix

| Release | Board controller (Recommended) | Board controller (Supported) | Flashed Firmware (Recommended) | Flashed Firmware (Supported) | PCIe driver – Linux (Recommended) | PCIe driver – Linux (Supported) | PCIe driver – Windows (Recommended) | PCIe driver – Windows (Supported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1.6.0 | 7.4 | 7.0 | 1.6.0 | 1.5.0, 1.4.0 | 1.4.16 | 1.4.10, 1.4.4 | 1.3.4 | 1.3.1, 1.3.0 |
| v1.6.1 | 7.4 | 7.0 | 1.6.0 | 1.5.0, 1.4.0 | 1.4.17 | 1.4.16, 1.4.10, 1.4.4 | 1.3.11 | 1.3.4, 1.3.1, 1.3.0 |
| v1.7.0 | 7.4 | 7.0 | 1.7.0 | 1.6.0, 1.5.0, 1.4.0 | 1.5.5 | 1.5.5 | 1.3.11 | 1.3.5, 1.3.4, 1.3.1, 1.3.0 |
| v1.8.0 | 7.4 | 7.0 | 1.8.0 | 1.7.0, 1.6.0, 1.5.0, 1.4.0 | 1.6.2 | 1.6.2 | 1.3.14 | 1.3.14 |

## Release-specific notes

### v1.8.0

The minimum *supported* driver version is equal to the shipped version, so **no earlier driver is accepted**:

- Linux: install `metis-dkms` **1.6.2**
- Windows: install **MetisDriver-1.3.14**

Board controller firmware requires **no** update in this release; the recommended version remains 7.4.

### v1.7.0

Metis M.2 Max is required to be updated to the recommended board controller and firmware versions. The `axdevice interactive_flash_update` script handles board variant selection automatically.

## Further Support

- For blog posts, projects and technical support please visit the [Axelera AI Community](https://community.axelera.ai/).
- For technical documents and guides please visit [docs.axelera.ai](https://docs.axelera.ai/).
