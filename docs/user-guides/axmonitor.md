---
title: "axmonitor"
---
# axmonitor

`axmonitor` provides real-time visibility into Metis device metrics — temperatures, utilization, memory, power, and FPS — similar to how `htop` or `nvidia-smi` work for CPUs and GPUs.

---

## What it monitors

| Metric | Description |
|--------|-------------|
| Core utilization | CPU usage per core and thread |
| Temperatures | Per-core and board temperatures with thermal threshold indicators |
| Kernels Per Second (KPS) | Number of fused AI kernels executed per second (≈ FPS for vision pipelines) |
| Power | Real-time power consumption (Supported on Metis PCIe rev2, M.2 Max, PCIe 4-AIPU and Metis Compute Boards) |
| DDR memory | Memory allocation per active application context |
| PCIe bandwidth | DMA bandwidth per channel in MB/s |
| Processes | Running processes with PIDs and container IDs (Linux only) |
| Device version | Firmware version, hardware ID, protocol version |
| Device configuration | Thermal management settings, clock frequency per core |
| MVM utilization | Percentage of the MVM (Matrix-Vector Multiplier) array that is active per core |
| Core stack usage | Stack memory utilization per core |

---

## Architecture

`axmonitor` is a frontend that connects to `axsystemserver`, a backend service that collects and streams device metrics over TCP. You must start the backend before running the tool.

```
axsystemserver  ←── collects device metrics
      ↕  TCP (default: 127.0.0.1:5555)
 axmonitor     ──── displays metrics
```

---

## Setup

### Start the backend service

**Linux:**
```bash
sudo systemctl start axsystemserver.service
```

Verify it's running:
```bash
sudo systemctl status axsystemserver.service
```

**Linux (Docker):**
```bash
sudo service axsystemserver start
sudo service axsystemserver status
```

**Windows** (install as Windows Service first, from an admin console):
```
axsystemserver --install-service
sc.exe start axsystemserver
```

Verify:
```
sc.exe query axsystemserver   # look for STATE: RUNNING
```

### Installing after pip installation

> [!NOTE]
> If you followed the [pip installation guide](sdk-install.md), install the base package before using axmonitor. If you used the SDK installer, skip this step.


```bash
# Add the Axelera apt repository (if not already configured)
sudo sh -c "curl -fsSL https://software.axelera.ai/artifactory/api/security/keypair/axelera/public | gpg --dearmor -o /etc/apt/keyrings/axelera.gpg"
# Ubuntu 22.04
sudo sh -c "echo 'deb [signed-by=/etc/apt/keyrings/axelera.gpg] https://software.axelera.ai/artifactory/axelera-apt-source ubuntu22 main' > /etc/apt/sources.list.d/axelera.list"
# Ubuntu 24.04
sudo sh -c "echo 'deb [signed-by=/etc/apt/keyrings/axelera.gpg] https://software.axelera.ai/artifactory/axelera-apt-source ubuntu24 main' > /etc/apt/sources.list.d/axelera.list"

sudo apt-get update
sudo apt-get install -y axelera-voyager-sdk-base-1.6.0
source /opt/axelera/sdk/1.6.0/axelera_activate.sh
```

### Run axmonitor

> [!TIP]
> Make sure the SDK virtual environment is activated (`source venv/bin/activate`) before running `axmonitor`.


```bash
axmonitor --server-address "127.0.0.1:5555"
```

This connects to the backend and opens the monitoring interface. Metrics update every second.

---

## Interface modes

### GUI (default)

The GUI organizes metrics into three sections:
- **OVERVIEW** — dashboard with key metrics and visualizations
- **SYSTEM** — process monitoring, I/O metrics
- **DEVICE** — per-device detail: CPU, memory, temperature, power, firmware, configuration

### Console mode

For SSH sessions or scriptable environments:

```bash
axmonitor --server-address "127.0.0.1:5555" --ui console
```

Interactive commands at the `axmonitor>` prompt:

| Command | Description |
|---------|-------------|
| `print` | Print current measurements |
| `help` | List available commands |
| `exit` | Quit |

### One-shot query

```bash
axmonitor --server-address "127.0.0.1:5555" --ui console -c print
```

Prints current metrics and exits — useful in scripts.

---

## Multi-device monitoring

To monitor specific devices:

```bash
axmonitor --server-address "127.0.0.1:5555" --topics DEV0 DEV1
```

Topics are `DEV0`, `DEV1`, ... `DEVN` where N is the number of connected devices. Default: `DEV0`.

---

## Changing the service port

If port 5555 is in use, reconfigure `axsystemserver`.

**Linux** — edit `/lib/systemd/system/axsystemserver.service`:
```
ExecStart=/bin/bash -c 'exec axsystemserver --bind "127.0.0.1:5556"'
```
Then restart: `sudo systemctl restart axsystemserver.service`

**Windows** — reinstall with a different port:
```
sc.exe delete axsystemserver
axsystemserver --install-service --bind 127.0.0.1:5556
```

---

## Recording and export

`axmonitor` can record metrics to disk and export them in human-readable or structured formats for offline analysis.

### Record a session

Use `--record` to save all received measurements to a `.raw` binary file:

```bash
axmonitor --server-address "127.0.0.1:5555" --record
```

By default the file is written to `axmonitor.data.raw`. Use `--data-file` to change the base path:

```bash
axmonitor --server-address "127.0.0.1:5555" --record --data-file my_session
```

### Export to text formats

Use `--export` together with `--record` to write metrics alongside the `.raw` file:

| Format | Extension | Description |
|--------|-----------|-------------|
| `jsonl` | `.jsonl` | Structured JSON Lines for scripting and post-processing |
| `pretty` | `.pretty` | Human-readable text output |

Multiple formats can be specified at once:

```bash
axmonitor --server-address "127.0.0.1:5555" --record --export jsonl pretty
```

This produces `axmonitor.data.raw`, `axmonitor.data.jsonl`, and `axmonitor.data.pretty`.

### Replay a recorded session

Use `--replay` to replay a previously recorded `.raw` file through the normal UI without a live device:

```bash
axmonitor --replay --data-file my_session
```

### Post-processing

axmonitor includes a post-processing analysis functionality with automated plot generation and statistical reporting, based on recorded device data saved in JSON Lines file (.jsonl).

The post-processor generates a folder (named <file.jsonl>_postproc) containing:
- 1 SVG file containing multi-panel grid plots (temperature, power, KPS, DDR bandwidth, PCIe bandwidth, CPU utilisation) for each device
- 1 SVG file containing dual-axis temperature vs total device KPS correlation plot for each device
- 1 SVG file containing dual-axis average power vs device KPS plot for each device (if power sensors are present)
- Markdown summary report for all devices

Users might want to focus analysis on a specific time window (e.g., seconds 300–500) rather than the full recording. The --post-proc-start and --post-proc-end CLI flags trim the data to the requested window (in seconds) before any plot or report is generated.

Post-processing invocation:

Full run:
```bash
axmonitor --post-proc <file.jsonl>
```

Run on a specific time window:
```bash
axmonitor --post-proc <file.jsonl> --post-proc-start <start-time> --post-proc-end <stop-time>
```

The post-processing runs in standalone mode and exits immediately.

---

## Warnings and threshold events

`axmonitor` displays current temperatures and highlights active thermal thresholds visually, but does **not** emit event-based messages when thresholds are crossed. Threshold crossing events — such as software throttling activating or frequency downscaling engaging — are only visible in the device firmware log stream.

To capture these events in real time, stream the firmware log using `axlogdevice`:

```bash
axlogdevice --slog-level inf:service_monitor
axlogdevice --slog
```

> [!NOTE]
> These events are not surfaced in `axmonitor` output or in `axmonitor --export` data files. If you need to correlate thermal events with performance data, run `axlogdevice` alongside `axmonitor` in a separate terminal.


---

## Troubleshooting

**No data displayed** — the most common cause is `axsystemserver` not running or a port conflict.

Check if the service is running (see [Setup](#setup) above). Check if the port is in use:

```bash
# Linux
sudo lsof -i :5555

# Windows
netstat -ano | findstr :5555
```

If the port is occupied, change the service configuration to a free port, then launch `axmonitor` with `--server-address` pointing to the new port.

**Check if messages are arriving:**
```bash
axmonitor --server-address "127.0.0.1:5555" -l info
```

If the log shows no messages being received, the backend isn't running.

---

## See also

- [Thermal Management](thermal.md) — temperature thresholds and throttling
- [Performance Metrics](../reference/measurement/performance.md) — FPS measurement during inference
- [axdevice](../reference/tools/axdevice.md) — check device detection and firmware version
