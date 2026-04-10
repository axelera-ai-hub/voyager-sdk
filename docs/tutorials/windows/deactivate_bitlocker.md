![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Deactivating BitLocker in 3 steps

## Contents
- [Deactivating BitLocker in 3 steps](#deactivating-bitlocker-in-3-steps)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Level](#level)
  - [Content](#content)
  - [Step 1: Open BitLocker](#step-1-open-bitlocker)
  - [Step 2: Deactivate BitLocker](#step-2-deactivate-bitlocker)
    - [Scenario 1: BitLocker is Not Active](#scenario-1-bitlocker-is-not-active)
    - [Scenario 2: BitLocker is Active](#scenario-2-bitlocker-is-active)
  - [Step 3: Confirm Deactivation](#step-3-confirm-deactivation)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Further support](#further-support)

---
## Prerequisites
- Windows 11 with administrative privileges
- BitLocker recovery key (if BitLocker is active)

## Level
**Beginner** - Simple Windows system configuration

## Content
- [Deactivating BitLocker in 3 steps](#deactivating-bitlocker-in-3-steps)
  - [Step 1: Open BitLocker](#step-1-open-bitlocker)
  - [Step 2: Deactivate BitLocker](#step-2-deactivate-bitlocker)
    - [Scenario 1: BitLocker is Not Active](#scenario-1-bitlocker-is-not-active)
    - [Scenario 2: BitLocker is Active](#scenario-2-bitlocker-is-active)
  - [Step 3: Confirm Deactivation](#step-3-confirm-deactivation)
  - [Next Steps](#next-steps)

This guide explains how to temporarily deactivate BitLocker **for the duration of one reboot** on a Windows system. BitLocker needs to be deactivated during the driver installation process to ensure proper system access.

## Step 1: Open BitLocker
1. Open the Windows search bar
2. Type "BitLocker"
3. Select "Manage BitLocker" from the search results

![Open BitLocker](/docs/images/windows/start_bitlocker.png)

## Step 2: Deactivate BitLocker
Look at the screen. One of two scenarios will be visible:

### Scenario 1: BitLocker is Not Active
If this screen is visible, BitLocker is already deactivated:
![BitLocker Deactivated](/docs/images/windows/bitlocker_deactivated.png)

In this case, proceed with the installation steps from the [Considerations](/docs/tutorials/windows/installing_driver.md#considerations) section.

### Scenario 2: BitLocker is Active
If BitLocker is active, this screen will be visible:
![Suspend Protection](/docs/images/windows/suspend_protection.png)

Click "Suspend protection" to continue.

## Step 3: Confirm Deactivation
A confirmation prompt will appear. Click "Yes" to confirm the BitLocker deactivation:

![Confirm Deactivation](/docs/images/windows/bitlocker_disable_prompt.png)

> [!WARNING]
> BitLocker will automatically reactivate after the next system reboot. If additional changes using `bcdedit` are needed in the future, follow these steps again to temporarily deactivate BitLocker.

## Next Steps
- **After deactivation**: Proceed to [Installing Driver](installing_driver.md#considerations) and start from the consideration section.
- **After driver installation**: Re-enable BitLocker for security
- **Continue setup**: [Windows Getting Started](windows_getting_started.md)

## Related Documentation
**Windows Guides:**
- [Installing Driver](installing_driver.md) - Driver installation REQUIRES BitLocker deactivation
- [Windows Getting Started](windows_getting_started.md) - Complete Windows setup guide

## Further support

For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).

For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).
