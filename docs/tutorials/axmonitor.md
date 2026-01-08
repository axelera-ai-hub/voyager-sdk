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
    - [2. Run axmonitor](#2-run-axmonitor)
    - [axmonitor Command-Line Options](#axmonitor-command-line-options)
  - [User Interface modes](#user-interface-modes)
    - [GUI](#gui)
    - [Console mode](#console-mode)
      - [Interactive Commands](#interactive-commands)
    - [One-shot Command Execution (-c option)](#one-shot-command-execution--c-option)
    - [Topics](#topics)
      - [Device Topics](#device-topics)
  - [Communication Architecture](#communication-architecture)
  - [Notes](#notes)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Further support](#further-support)

## Prerequisites
- SDK installed and virtual environment activated
- Hardware connected and detected (`axdevice`)
- `axsystemserver` service running (automatically starts with SDK)

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
  -l [ --log-level ] arg (=error)       Logging level (trace, debug, info, 
                                        warning, error, fatal)
  --log-file arg                        Redirect log messages to a file
  --server-address arg (=127.0.0.1:5555)
                                        Server address to connect to for 
                                        receiving metrics, in the format 
                                        IP:Port (default: 127.0.0.1:5555)
  --ui arg (=auto)                      ui mode: auto, console, gui
  -c [ --command ] arg                  Commands are read from string
  -t [ --topics ] arg                   List of topics to subscribe to
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

## Notes

- axsystemservice must be running in the background before starting axmonitor.
- Future releases may expand monitored metrics.

## Next Steps
- **Monitor during benchmarks**: Run axmonitor while executing [Benchmarking Tutorial](benchmarking.md)
- **Understand thermal behavior**: Read [Thermal Guide](../reference/thermal_guide.md) for threshold details
- **Optimize performance**: Use metrics to identify bottlenecks in your application

## Related Documentation
**Tutorials:**
- [Benchmarking](benchmarking.md) - Use axmonitor during benchmark runs to track resource usage
- [Application Integration](application.md) - Monitor your applications in real-time

**References:**
- [Thermal Guide](../reference/thermal_guide.md) - Understanding temperature thresholds and thermal management
- [AxDevice API](../reference/axdevice.md) - Programmatic device enumeration and management

## Further support
- For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).
- For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).
