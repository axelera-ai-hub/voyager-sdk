---
title: "Firmware Update"
---
# Firmware Update

How to update the firmware on your Metis PCIe or M.2 board.

> [!WARNING]
> **Firmware updates are currently Linux-only.** If you are on Windows, temporarily connect the board to a Linux host to perform the update, then reconnect to Windows when done.


---

## Prerequisites

- Voyager SDK installed and virtual environment activated (`source venv/bin/activate`)
- Administrative/sudo privileges
- Uninterrupted power supply recommended for board controller updates (see [Safety](#safety))

---

## Flash firmware

> [!CAUTION]
> **DO NOT interrupt the flash process**
> Once the update starts, **DO NOT**:
> - Power off the system
> - Close the terminal or quit the script
> - Run system updates or reboot
> - Let the device overheat (thermal shutdown during Stage 2 is unrecoverable)
>
> Use a UPS where possible. See [Safety](#safety) below for stage-by-stage risk details.


### Single device

Run the interactive update script:

```bash
axdevice interactive_flash_update
```

Follow the on-screen prompts. The script will tell you when a power cycle is needed. Re-run the script after each power cycle until it completes.

> [!WARNING]
> Do not modify the `interactive_flash_update` procedure. It contains safety checks that must not be altered.


### Multiple devices

**Option A — automatic (recommended):** Run the same script without specifying a device. It detects all connected cards and updates them all. Re-run after each required power cycle until all devices are updated:

```bash
axdevice interactive_flash_update
# Re-run after each power cycle until: "ALL DEVICES ARE FULLY UPDATED!"
```

**Option B — one at a time:** First list device IDs:

```bash
axdevice
# Device 0: metis-0:6c:0 ...
# Device 1: metis-0:6d:0 ...
```

Then update each device individually:

```bash
axdevice interactive_flash_update --device metis-0:6c:0
axdevice interactive_flash_update --device metis-0:6d:0
# ... repeat for each device
```

---

## Verify the update

After the final power cycle:

```bash
axdevice
```

Expected output:
```
Device 0: metis-0:1:0 4GiB pcie flver=1.4.0 bcver=7.0 clock=800MHz(0-3:800MHz) mvm=0-3:100%
```

Check `flver` (firmware version) and `bcver` (board controller version) match the target release.

---

## Safety

The firmware update process has two stages with different risk profiles:

| Stage | What gets updated | Recovery if power fails | Risk |
|-------|-------------------|------------------------|------|
| Stage 1 | Metis firmware | Dual-boot regions — previous version still available | Low |
| Stage 2 | Board controller firmware | No failsafe — single region | **High** |

For Stage 2, ensure:
- Use a UPS if possible
- No system updates or reboots during the process
- Close unnecessary applications
- Monitor temperature — a thermal shutdown during Stage 2 is unrecoverable

If Stage 2 fails, contact Axelera AI support.

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| Script fails at start | Confirm `source venv/bin/activate` is active |
| Board not detected after update | Confirm full power off was done (not just reboot) |
| Errors during script | Review terminal output for error messages; do not retry a partial Stage 2 |

---

## See also

- [axdevice](../reference/tools/axdevice.md) — verify device detection and firmware version
- [SDK Installation](sdk-install.md) — SDK must be installed before flashing
