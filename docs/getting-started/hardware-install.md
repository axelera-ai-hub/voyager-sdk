# Hardware Installation

This page covers physically installing your Metis hardware. Choose your product below.

> [!WARNING]
> Disconnect the power source before handling or installing any hardware. Wear an anti-static wrist strap to prevent electrostatic discharge (ESD). Do not touch the connector pins.


## Metis M.2 AI Accelerator

### Prerequisites

- An M.2 **2280 M-key** slot (not B-keyed, not SATA-based)
- The slot must provide up to 11.55W average power (peak: 23.1W)
- Ambient temperature range: -20°C to 70°C
- Anti-static wrist strap


### Install the card

1. Power down your system and disconnect it from the power source.
2. Locate an available M.2 2280 M-key slot on your motherboard.
3. Align the card with the slot and insert it at a 20-30 degree angle.
4. Push the card down until it lies flat against the motherboard.
5. Secure it using the bracket screw or M.2 standoff screw.

### Verify cooling

- If using the **active cooling** model: confirm the fan is securely attached and unobstructed.
- If using the **passive cooling** model: confirm the heatsink has adequate ventilation.
- Ensure the system chassis has sufficient airflow for heat dissipation.

### Verify power

Run this command to check PCIe power availability:

```bash
sudo lspci -vvv | grep -i "LnkSta" -A 10
```

Look for `MaxPower` in the output. The slot must supply at least 11.55W average (23.1W peak).

If insufficient power is detected, check your motherboard specifications or consider an alternative carrier board.

### Post-install checklist

- [ ] Card is fully inserted and secured with a screw
- [ ] Cooling is functioning (fan spinning or heatsink ventilated)
- [ ] System chassis has adequate airflow
- [ ] System detects the card: `lspci | grep Axelera`

---

## Metis PCIe 1-AIPU AI Accelerator Card

### Prerequisites

- A PCIe Gen 3.0 x4 slot (or higher)
- Minimum 25W through the PCIe slot (recommended: 75W)
- Ambient temperature range: -20°C to 70°C
- Anti-static wrist strap


### Install the card

1. Power down your system and disconnect it from the power source.
2. Locate an available PCIe Gen 3.0 x4 (or higher) slot.
3. Align the card and insert it firmly into the slot.
4. Secure the card with the bracket screws.

### Verify cooling

- Confirm the fan is securely attached and unobstructed.
- Ensure the system chassis has adequate airflow.

### Verify power

```bash
sudo lspci -vvv | grep -i "LnkSta" -A 10
```

Look for `MaxPower` in the output. If your system does not meet the recommended 75W, check your motherboard or power supply specifications.

### Post-install checklist

- [ ] Card is firmly seated in the PCIe slot
- [ ] Bracket screws are secured
- [ ] Fan is running and unobstructed
- [ ] System detects the card: `lspci | grep Axelera`

---

## Metis PCIe 4-AIPU AI Accelerator Card

### Prerequisites

- A PCIe Gen 3.0 **x16** slot
- An 8-pin PCIe auxiliary power cable from your power supply
- Anti-static wrist strap

### Install the card

1. Align the card with a PCIe Gen 3.0 x16 slot and insert it.
2. Secure the card with the bracket screws.
3. Verify the fan is securely attached and unobstructed.
4. Connect the PCIe auxiliary power cable to the 2x4 connector on the card.

> [!NOTE]
> The Metis PCIe 4-AIPU card requires auxiliary power. It will not function from slot power alone.


### Post-install checklist

- [ ] Card is seated and bracket screws secured
- [ ] Auxiliary power cable connected (2x4 connector)
- [ ] Fan is running and unobstructed
- [ ] System detects the card: `lspci | grep Axelera`

---

## Metis Compute Board

The Compute Board is a standalone system — there is no card to install. Connect peripherals and power to get started.


### Connect the board

1. Connect the CPU and chassis fans to their respective fan headers.
2. (Optional) Connect Power and Reset switches to their headers.
3. Plug a mouse and keyboard into the USB ports.
4. Connect a LAN cable to the LAN port.
5. Attach an HDMI cable to the HDMI port and your display.
6. Connect the DC power cable to the DC Power In.

> [!NOTE]
> Use a 12V, 7A DC adapter with a barrel connector size of 5.5mm (outer) x 2.5mm (inner).


The board boots automatically when power is connected. It displays the Axelera logo during POST. Access a shell by clicking the terminal icon in the top-left corner.

### Compute Board: additional setup

The Compute Board requires flashing the Board Support Package (BSP) and runs the Voyager SDK in a Docker container. These steps differ from the PCIe/M.2 path.


> [!TIP]
> If your board came pre-installed with a recent BSP, you can skip straight to the SDK Docker setup in Section 5 of the User Guide.


---

## Next steps

Your hardware is installed. Proceed to [Install the SDK](../user-guides/sdk-install.md).
