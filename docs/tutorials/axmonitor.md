![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# axmonitor - Axelera AI System Monitoring Tool

`axmonitor` is a system monitoring tool for Axelera AI devices which runs on the host and provides real-time visibility into device metrics, much like CPU/GPU monitoring tools.

## Contents

- [axmonitor - Axelera AI System Monitoring Tool](#axmonitor---axelera-ai-system-monitoring-tool)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Level](#level)
  - [Overview](#overview)
  - [Usage](#usage)
    - [Windows Service Installation](#windows-service-installation)
    - [Configuring the Axelera System Service](#configuring-the-axelera-system-service)
      - [Linux](#linux)
      - [Windows](#windows)
    - [1. Start the System Monitoring Backend Service](#1-start-the-system-monitoring-backend-service)
      - [Linux](#linux-1)
      - [Windows](#windows-1)
      - [Verify the Service is Running](#verify-the-service-is-running)
    - [2. Run axmonitor](#2-run-axmonitor)
    - [axmonitor Command-Line Options](#axmonitor-command-line-options)
  - [User Interface modes](#user-interface-modes)
    - [GUI](#gui)
    - [Console mode](#console-mode)
      - [Interactive Commands](#interactive-commands)
    - [One-shot Command Execution (-c option)](#one-shot-command-execution--c-option)
    - [Topics](#topics)
      - [Device Topics](#device-topics)
  - [Warnings and Threshold Events](#warnings-and-threshold-events)
  - [Recording and Export](#recording-and-export)
    - [Recording](#recording)
    - [Export](#export)
    - [Replay](#replay)
  - [Communication Architecture](#communication-architecture)
  - [Troubleshooting](#troubleshooting)
    - [No Data Displayed in axmonitor](#no-data-displayed-in-axmonitor)
  - [Notes](#notes)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Further support](#further-support)

## Prerequisites
- SDK installed and virtual environment activated
- Hardware connected and detected (`axdevice`)
- `axsystemserver` service running (automatically starts with SDK)

### Installing after pip installation guide

> [!Note] In case you followed [Pip Installation Guide](./install_pip.md) follow the below steps.
> If you followed [SDK Installer guide](./install.md) you can skip these steps

```bash
# Step 1: Download and add the public key to the system keyring
sudo sh -c "curl -fsSL https://software.axelera.ai/artifactory/api/security/keypair/axelera/public | gpg --dearmor -o /etc/apt/keyrings/axelera.gpg"
# Step 2: Add the repository to apt list
# Ubuntu 22.04
sudo sh -c "echo 'deb [signed-by=/etc/apt/keyrings/axelera.gpg] https://software.axelera.ai/artifactory/axelera-apt-source ubuntu22 main' > /etc/apt/sources.list.d/axelera.list"
# Ubuntu 24.04
sudo sh -c "echo 'deb [signed-by=/etc/apt/keyrings/axelera.gpg] https://software.axelera.ai/artifactory/axelera-apt-source ubuntu24 main' > /etc/apt/sources.list.d/axelera.list"
# Step3: Update the list of packages
sudo apt-get update
sudo apt-get install -y axelera-voyager-sdk-base-1.6.0
source /opt/axelera/sdk/1.6.0/axelera_activate.sh
```

## Level
**Beginner** - Simple monitoring tool, no coding required

## Overview

The `axmonitor` application enables users to monitor key metrics from Axelera AI devices. It visualizes the current operational status of devices, including:

- **Core Utilization**: CPU usage per core and thread
- **Core & Board Temperatures**: Temperature readings from cores and board sensors with thermal threshold indicators (Frequency Downscale, SW Throttling, HW Throttling)
- **Kernels Per Second (KPS)**: Number of kernels executed per second on AI cores
  - Kernels on Axelera accelerators are always a number of fused compute operations together. They can refer to an inference or inference and a few mathematical functions combined. This parameter shows the number of these fused kernels executed per second on the hardware. For vision pipelines it is often the same as frames per second.
- **Power Usage**: Real-time power consumption from board sensors
- **DDR Memory Usage**: DDR utilization per context in MB, showing memory allocation across active application contexts
- **DDR Memory Bandwidth**: Real-time DDR memory bandwidth in GB/s, displaying read, write, and total bandwidth utilization
- **PCIe Bandwidth**: PCIe DMA bandwidth per channel in MB/s
- **Process Monitoring**: Running processes with PIDs, arguments, and container IDs (Linux only)
- **Timestamps**: System and device timestamps in milliseconds
- **Device Version**: Protocol version, firmware version, SOM firmware version, and hardware ID
- **Device Configuration**: Thermal management settings (temperature thresholds, hysteresis values, MVM limits) and clock frequency per core

These metrics are retrieved periodically (every 1 second) over a TCP/IP network connection to the Axelera System Service (`axsystemservice`), a host system service for device monitoring and management.

> [!NOTE]  
> Power measurements are supported only on [4-Metis AIPU PCIe cards](https://store.axelera.ai/products/pcie-ai-accelerator-card-powered-by-4-metis-aipu).

## Usage

Before running `axmonitor`, make sure the Axelera System Service (`axsystemserver`) is active. This service exposes a server that streams
real-time device metrics, which `axmonitor` connects to as a subscriber.

### Windows Service Installation

If working on Windows, you can install the Axelera System Service as a Windows Service. To do so, run the following command from an
administrator console:
```
axsystemserver --install-service [--bind IP:PORT]
```

The command-line arguments used together with `--install-service` will persist in the service configuration. The server will start with
those arguments by default.

### Configuring the Axelera System Service

By default, the Axelera System Service listens on `*:5555`. If you need to change this IP address or port, you can do so.

#### Linux

To change the default listen IP address or port on Linux, modify the `/lib/systemd/system/axsystemserver.service` file.
Locate and update the following line:
```
ExecStart=/bin/bash -c 'exec axsystemserver --bind "<IP>:<Port>"'
```
> [!NOTE]
> If you're running inside a docker container, modify `/etc/init.d/axsystemserver` instead. Find and adjust the following line:
> ```
> DAEMON_ARGS="--bind <IP>:<Port>"
> ```

Replace `<IP>:<Port>` with your desired address and port.

#### Windows

If you [installed](#windows-service-installation) the Axelera System Service as a Windows Service and want to modify the default listen
IP address or port, delete and re-install the Windows service with the desired IP and port in the `--bind` argument.

From an admin console, you can run:
```
sc.exe delete axsystemserver
axsystemserver --install-service --bind IP:PORT
```

### 1. Start the System Monitoring Backend Service

This launches the backend service, opening the configured TCP address/port for device metrics streaming.

#### Linux

To start the Axelera System Service on Linux run:
```
sudo systemctl start axsystemserver.service
```
> [!NOTE]
> If you're running inside a docker container, you can use the init.d script instead:
> ```
> sudo service axsystemserver start
> ```

#### Windows

If you [installed](#windows-service-installation) the Axelera System Service as a Windows Service, run the following command from
an administrator console to start it:
```
sc.exe start axsystemserver
```

If you want to start the service with other arguments compared to those used during service installation, you can pass the new arguments
to the `sc.exe` command. E.g.:
```
sc.exe start axsystemserver --bind IP:PORT
```

Alternatively, you can start or stop the service by using the Services application, which you can access by searching for "services" and
opening it, or by using the `services.msc` command in the Run dialog.

#### Verify the Service is Running

After starting the service, verify that it has launched successfully.

**Linux:**

Check the service status:
```bash
sudo systemctl status axsystemserver.service
```

If the service is running correctly, you should see an `active (running)` status. If there are any issues, you can view detailed logs:
```bash
sudo journalctl -u axsystemserver.service
```

> [!NOTE]
> If you're running inside a docker container, use `sudo service axsystemserver status` instead.

**Windows:**

You can verify the service status using the Services application (`services.msc`) or by running:
```
sc.exe query axsystemserver
```

Look for the `STATE` field showing `RUNNING`. If there are issues, check the Windows Event Viewer for error logs.

If the service is not running or you encounter errors, refer to the [Troubleshooting](#troubleshooting) section for common issues and solutions.

### 2. Run axmonitor

Once the service is running, start axmonitor by specifying the subscription address:

`axmonitor --server-address "127.0.0.1:5555"`

`axmonitor` will connect to the service’s TCP endpoint, receive device measurements every 1 second, and display them in a structured monitoring interface.

### axmonitor Command-Line Options

`axmonitor` provides a set of command-line options to customize its behavior. To view available options:
```
axmonitor --help
```
Example Help Output:
```
Axelera system monitoring tool for real-time hardware accelerator metrics.

Usage: axmonitor [options]
Allowed options:

Global options:
  -h [ --help ]                         Produce help message
  --server-address arg (=127.0.0.1:5555)
                                        Server address to connect to for 
                                        receiving metrics, in the format 
                                        IP:Port (default: 127.0.0.1:5555)
  --ui arg (=auto)                      ui mode: auto, console, gui
  -c [ --command ] arg                  Commands are read from string
  -t [ --topics ] arg                   List of topics to subscribe to
  --record                              Enable recording of received data to a
                                        .raw file (always needed for replay)
  --replay                              Enable replaying data from a recorded
                                        .raw file
  --data-file arg (=axmonitor.data)     Base path for data files ('.raw' and/or
                                        '.json' will be appended)
  --export arg                          Export formats: json (structured),
                                        pretty (human-readable). Multiple
                                        allowed.
```

## User Interface modes

### GUI

The axmonitor GUI organizes metrics into a tree-structured interface with multiple pages. The main sections include:

- **OVERVIEW**: Dashboard with key system metrics and real-time visualizations
- **SYSTEM**: System-level information including process monitoring (Linux only) and I/O metrics
- **DEVICE**: Detailed per-device monitoring with subsections for CPU, memory, temperature, power, version, and configuration

### Console mode

In addition to its graphical user interface, `axmonitor` also supports a console mode designed for terminal-based interaction. This mode is ideal for users who prefer lightweight or scriptable environments.

To start axmonitor in console mode:

```bash
axmonitor --server-address "127.0.0.1:5555" --ui "console"
```

Upon launch, you will see a prompt similar to:

```
axmonitor> Welcome to axmonitor. Type 'help' for available commands.
```

#### Interactive Commands

Once in console mode, you can enter commands directly at the prompt. Currently supported commands include:

```
axmonitor> help
Commands available:
 - help
        This help message
 - exit
        Quit the session
 - print
        Print last measurements
```

### One-shot Command Execution (-c option)

Console mode also supports non-interactive execution using the `-c` option. The provided command string must be a valid console command (e.g., `print`, etc.). This behaves similarly to a typical shell command - the specified command is executed immediately, and the tool exits. 

Example:

```bash
axmonitor --server-address "127.0.0.1:5555" --ui "console" -c print
```

This is especially useful for scripting or automation scenarios where only a single metric query or action is needed.

### Topics

In axmonitor, topics are used to filter the device metrics being subscribed to. 

#### Device Topics

These topics correspond to connected devices, allowing users to choose which specific device's metrics they want to monitor.

The available topics are:

- `DEV0`, `DEV1`, ..., `DEVN` where `N` is the number of connected devices. By default, the first device (`DEV0`) is chosen if no topic is specified.

To subscribe to a specific device, use the `--topics` option followed by a space-separated list of topics:

```bash
axmonitor --server-address "127.0.0.1:5555" --topics DEV0 DEV1
```

This will restrict the monitoring to only the devices `DEV0` and `DEV1`.

## Warnings and Threshold Events

`axmonitor` displays current temperatures and highlights active thermal thresholds visually, but does not emit event-based messages when thresholds are crossed. Threshold crossing events, such as SW throttling activating or frequency downscaling engaging, are only visible in the device firmware log stream.

To capture these events in real time, stream the firmware log using `axlogdevice`:

```bash
axlogdevice --slog-level inf:service_monitor
axlogdevice --slog
```

> [!NOTE]
> These events are not surfaced in `axmonitor` output or in the `axmonitor --export` data files. If you need to correlate thermal events with performance data, run `axlogdevice` alongside `axmonitor` in a separate terminal.

## Recording and Export

`axmonitor` can record incoming metrics to disk and export them in human-readable or structured formats for offline analysis.

### Recording

Use `--record` to save all received measurements to a `.raw` binary file:

```bash
axmonitor --server-address "127.0.0.1:5555" --record
```

By default the file is written to `axmonitor.data.raw`. Use `--data-file` to change the base path:

```bash
axmonitor --server-address "127.0.0.1:5555" --record --data-file my_axmonitor_session
```

This writes to `my_axmonitor_session.raw`.

### Export

Use `--export` together with `--record` to write metrics in one or more text formats alongside the `.raw` file:

| Format   | Output file extension | Description                        |
|----------|-----------------------|------------------------------------|
| `json`   | `.json`               | Structured JSON, suited for scripting and post-processing |
| `pretty` | `.pretty`             | Human-readable text output         |

Multiple formats can be specified at once:

```bash
axmonitor --server-address "127.0.0.1:5555" --record --export json pretty
```

This produces `axmonitor.data.raw`, `axmonitor.data.json`, and `axmonitor.data.pretty`.

### Replay

Use `--replay` to replay a previously recorded `.raw` file through the normal UI without a live device or server connection:

```bash
axmonitor --replay --data-file my_axmonitor_session
```

## Communication Architecture

axmonitor acts as a TCP client, subscribing to a specific address/port opened by axsystemservice. This connection is used to fetch live data every second, ensuring an up-to-date monitoring experience.

**Service-Tool Interaction**
```
+----------------+         TCP         +---------------+
| axsystemservice | <----------------> |   axmonitor    |
+----------------+                     +---------------+
```

- axsystemservice: Backend service collecting and broadcasting metrics
- axmonitor: Frontend CLI/GUI displaying metrics

## Troubleshooting

### No Data Displayed in axmonitor

If `axmonitor` is running but you don't see any data being displayed, the most likely cause is that the `axsystemserver` service is not started, or it failed to start. This can happen for a number of reasons, the most common one being that the port is already in use by another application or service.

**Verify the service is running:**

First, check if the service is running and review the logs if needed. See the [Verify the Service is Running](#verify-the-service-is-running) section for instructions on how to check the service status and logs for both Linux and Windows.

**Check if axmonitor is receiving messages:**

- **GUI mode**: if the GUI shows no data or metrics are not updating, the tool is not receiving messages from the service.
- **Console mode**: run `axmonitor` with `--ui console` and use the `print` command. If measurements are shown, the tool is receiving data from the service.

If neither mode shows data after a few seconds, the service is either not running or failed to start.

**Check if the port is in use:**

If the service failed to start, the most common cause is a port conflict. You can check if the port is already in use:

**Linux:**
```bash
sudo lsof -i :<PORT>
```
Or alternatively:
```bash
sudo netstat -tulpn | grep <PORT>
```

**Windows:**
```
netstat -ano | findstr :<PORT>
```

Replace `<PORT>` with the port number you're trying to use (e.g., 5555 for the default). If the command returns any output, the port is already in use by another process.

**Solution:**

Change the address/port configuration to use a different available port. Refer to the [Configuring the Axelera System Service](#configuring-the-axelera-system-service) section for instructions on how to change the address and port for both Linux and Windows.

After changing the configuration, restart the `axsystemserver` service:

**Linux:**
```bash
sudo systemctl restart axsystemserver.service
```

**Windows:**
```
sc.exe stop axsystemserver
sc.exe start axsystemserver
```

Then:
1. [Verify the service is running](#verify-the-service-is-running) with the new settings
2. Launch `axmonitor` with the new address: `axmonitor --server-address "IP:PORT"`

## Notes

- axsystemservice must be running in the background before starting axmonitor.
- Future releases may expand monitored metrics.

## Next Steps
- **Monitor during benchmarks**: Run axmonitor while executing [Benchmarking Tutorial](benchmarking.md)
- **Understand thermal behavior**: Read [Thermal and Power Guide](../reference/thermal_and_power_guide.md) for threshold details
- **Optimize performance**: Use metrics to identify bottlenecks in your application

## Related Documentation
**Tutorials:**
- [Benchmarking](benchmarking.md) - Use axmonitor during benchmark runs to track resource usage
- [Application Integration](application.md) - Monitor your applications in real-time

**References:**
- [Thermal and Power Guide](../reference/thermal_and_power_guide.md) - Understanding temperature thresholds and thermal management
- [AxDevice API](../reference/axdevice.md) - Programmatic device enumeration and management

## Further support

For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).

For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).
