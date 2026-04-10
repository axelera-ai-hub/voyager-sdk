![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Installing the Windows Driver

## Contents
- [Installing the Windows Driver](#installing-the-windows-driver)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Level](#level)
  - [Summary](#summary)
  - [Considerations](#considerations)
  - [Installation steps](#installation-steps)
    - [Step 1 - Getting the Drivers](#step-1---getting-the-drivers)
    - [Step 2 - Installing the Drivers](#step-2---installing-the-drivers)
  - [Related Documentation](#related-documentation)
  - [Next Steps](#next-steps)
  - [Further support](#further-support)

## Prerequisites
- Windows 11 with administrative privileges
- BitLocker must be temporarily disabled (see link below)
- Internet connection to download driver files

## Level
**Beginner** - Follow step-by-step Windows driver installation

## Summary

This guide covers the installation of the Windows driver for Axelera Metis devices, and for the Switchtec PCIe switch (only if working with [4-Metis PCIe board](https://store.axelera.ai/collections/ai-acceleration-cards/products/pcie-ai-accelerator-card-powered-by-4-metis-aipu)).

## Considerations
As of October 2025, the Windows driver is not yet offered as a Microsoft-certified driver while the certification process is ongoing. Therefore, the driver is not installed automatically by Windows and needs to be installed manually. To enable manual installation, Windows needs to be set up in testsign:
 - Open a Windows Command Prompt in Administrator Mode
 - Set up testsigning like:
```bash
bcdedit /set testsigning on
```
  - Temporarily disable BitLocker - [See how here](/docs/tutorials/windows/deactivate_bitlocker.md)
  - Reboot the PC

The PC desktop should display "Test Mode" in the lower right corner.

> [!WARNING]  
> BitLocker will automatically reactivate after the next system reboot. If additional changes using `bcdedit` are needed in the future, for example, to run:
> ```bash
> bcdedit /set testsigning off
> ```
> follow [these steps](/docs/tutorials/windows/deactivate_bitlocker.md) again to temporarily deactivate BitLocker.
> **If BitLocker is not disabled, the system may go into Recovery mode and require contacting your network administrator for the harddrive recovery key on boot. It is very important to disable BitLocker when making bcdedit changes.**

## Installation steps
> [!NOTE]  
> These steps require access to a Windows Administrator account.
### Step 1 - Getting the Drivers
The driver archives can be downloaded from:
- [Metis-v1.3.4.zip](https://software.axelera.ai/artifactory/axelera-win/driver/1.3.x/MetisDriver-1.3.4.zip)
- [Switchtec-kmdf-0.7_2019.zip](https://software.axelera.ai/artifactory/axelera-win/driver/1.3.x/switchtec-kmdf-0.7_2019.zip) [Optional, only for 4-Metis PCIe board]

Then extract the drivers locally. Each extracted folder should contain three files: .cat, .inf and .sys.

### Step 2 - Installing the Drivers

Open the Device Manager by right clicking on the Windows start button and selecting Device Manager.

![Open Device Manager](/docs/images/windows/open_device_manager.png)

#### Removing a Previously Installed Driver

If you already have an older version of the Metis driver installed, you must fully remove it before installing the new one. Windows can keep multiple drivers for the same device, so it is important to completely uninstall the existing driver first.

In Device Manager, locate the Metis device under Neural Processors, right-click on it and select **Uninstall device**:

![Uninstall Device](/docs/images/windows/uninstall_device.png)

In the confirmation dialog, make sure to check the **Attempt to remove the driver for this device** checkbox, then click **Uninstall**:

![Remove Driver Checkbox](/docs/images/windows/uninstall_remove_driver.png)

After uninstallation, verify that the device now appears under **Other devices** as a generic **PCI Device**. This confirms the driver has been fully removed:

![PCI Device](/docs/images/windows/pci_device_other_devices.png)

You can now proceed with installing the new driver.

#### Installing the Drivers

From the menu, open Action and select Add Drivers:

![Add Driver](/docs/images/windows/add_driver.png)

For the Metis driver, set the path to the respective folder where the 3 driver files were extracted:

![Set Path](/docs/images/windows/set_path.png)

Select to install the driver anyway:

![Install Anyway](/docs/images/windows/install_anyway.png)

Confirm that the Metis device is now listed under Neural Processors:

![Metis Device](/docs/images/windows/metis_device.png)

If you operate a 4-Metis PCIe board, repeat the same steps to add the Switchtec driver as well.

![Switchtec Device](/docs/images/windows/switchtec_device.png)

## Related Documentation
**Windows Guides:**
- [Deactivate BitLocker](deactivate_bitlocker.md) - REQUIRED BEFORE driver installation
- [Windows Getting Started](windows_getting_started.md) - Complete Windows setup guide
- [Windows Voyager SDK Repository](windows_voyager_sdk_repository.md) - Repository setup

**General Guides:**
- [Installation Guide](../install.md) - Linux installation (alternative)
- [Quick Start Guide](../quick_start_guide.md) - After driver installation, run first inference

**References:**
- [AxDevice API](../../reference/axdevice.md) - Verify driver installation

## Next Steps
- **Re-enable BitLocker**: After successful driver installation
- **Continue Windows setup**: [Windows Getting Started](windows_getting_started.md)
- **Verify hardware detection**: Use AxDevice API
- **Run first inference**: [Quick Start Guide](../quick_start_guide.md)
---

- [Installing the Windows Driver](#installing-the-windows-driver)
  - [Summary](#summary)
  - [Considerations](#considerations)
  - [Installation steps](#installation-steps)
    - [Step 1 - Getting the Drivers](#step-1---getting-the-drivers)
    - [Step 2 - Installing the Drivers](#step-2---installing-the-drivers)

## Further support

For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).

For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).
