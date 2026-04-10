![](/docs/images/Ax_Page_Banner_2500x168_01.png)

# PIP Installation guide

> [!WARNING]
> This document is applicable for Axelera SDK versions >= 1.6

## Contents
- [Prerequisites](#prerequisites)
- [Level](#level)
- [Overview](#overview)
- [Install dependencies](#install-dependencies)
- [Install the Metis kernel driver](#install-the-metis-kernel-driver)
  - [System package manager](#system-package-manager)
  - [Axelera CLI command](#axelera-cli-command)
- [Python pip installation (recommended)](#python-pip-installation-recommended)
  - [Create a virtual environment](#create-a-virtual-environment)
  - [Install the full SDK environment](#install-the-full-sdk-environment)
  - [Verify installation](#verify-installation)
- [Slimmed down environments](#slimmed-down-environments)
  - [Runtime environment](#runtime-environment)
  - [Compiler environment](#compiler-environment)
- [Next Steps](#next-steps)
- [Related Documentation](#related-documentation)
- [Support](#support)
- [Further support](#further-support)

## Prerequisites
- Ubuntu 22.04+ or Windows with WSL2 installed
- Python 3.10, 3.11, 3.12, or 3.13
- Internet connection for downloading packages
- Administrative/sudo privileges (for kernel driver installation)
- PCIe slot for Metis hardware

## Level
**Beginner** - Follow step-by-step installation instructions

## Overview

The Voyager SDK is released in a GitHub repository. This repository contains
a branch for each publicly released version of the SDK. 
To checkout the repository, run the following command:


```bash
git clone https://github.com/axelera-ai-hub/voyager-sdk.git
```

This command downloads the repository in your current directory.

The default branch of the repository is always set to the latest published SDK release. You can use
standard `git` commands to list the available releases and to checkout different versions of the
SDK. Run the following command to  view the current release branch and all available releases:

```bash
git branch
```

To checkout a specific SDK release, run a command such as:

```
git checkout release/v1.6
git rebase
```

The latest publicly released SDK version is the default branch of the `git` repository; the name of
the branch can be found by visiting the Github page or by looking for `HEAD` at the output of 
`git remote show origin`. To rebase to the latest publicly released SDK version, run the following commands:

```bash
git fetch
export LATEST_RELEASE=$(git remote show origin | sed -n '/HEAD branch/s/.*: //p')
git checkout "${LATEST_RELEASE}"
git rebase
```

The Voyager SDK components are distributed via standard Python packages on Axelera-hosted pypi index, using `manylinux`-compatible wheels.

There are two installation paths:

| Path | Packages | Use case |
| :--- | :------- | :------- |
| **Python pip** (new) | `axelera-rt`, `axelera-devkit` | Standard deployment and development workflows |
| **SDK Installer**  | `install.sh` | Previous installation method |

This guide focuses on **Python pip** path. For the **SDK Installer** path, please refer to [SDK Installer guide](/docs/tutorials/install.md)

## Install dependencies

The Voyager SDK depends on several system-wide packages. Install them using the `install-dependencies.sh` script.

> [!NOTE]
> The `install-dependencies.sh` script requires `sudo` privileges


## Install the Metis kernel driver

The Metis PCIe driver is distributed as a DKMS Debian package and must be installed on every host system that connects to Metis hardware.

There are two ways to install the kernel driver: 
 - Through the system package manager
 - Through Axelera CLI command 

### System package manager
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
sudo apt-get install -y metis-dkms=1.4.16
```

> [!NOTE]
> The `metis-dkms` package requires a kernel build environment. On Ubuntu this is provided by the `linux-headers-$(uname -r)` package, which is typically installed by default. If the installation fails, ensure your kernel headers are present:
> ```bash
> sudo apt-get install -y linux-headers-$(uname -r)
> ```

After installation, verify the driver is loaded:

```bash
lsmod | grep metis
```

### Axelera CLI command

> [!WARNING] 
> This approach requires an activated virtual environment with Voyager SDK.
> Follow the steps presented in [Python pip installation (recommended)](#python-pip-installation-recommended)


The installation process of the Kernel driver is abstracted away by using `axdevice driver --install`. 

> [!NOTE]
> This route still requires `sudo` privileges.
> This route will **not** adapt APT source lists.


## Python pip installation (recommended)

This is the recommended installation path. It supports Python 3.10–3.13 and uses standard `manylinux`-compatible wheels, so no compiler or build toolchain is required on the host.

### Create a virtual environment

Create and activate a Python virtual environment before installing any Axelera packages:

```bash
python3 -m venv axelera-env
source axelera-env/bin/activate
```

> [!TIP]
> Use a dedicated virtual environment for each SDK version or project to avoid dependency conflicts.


> [!NOTE]
> You need the `-dev` and `-venv` variant for the Python version you want to use for the virtual environment
> These are not included in the `install-dependencies.sh` script.
> To install them, the most used command is `sudo apt-get install -y python3-dev python3-venv`


### Install the full SDK environment

```bash
# With the virtual environment activated
pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-rt axelera-devkit[all]
make operators
```

### Verify installation

After completing installation, verify that the Metis hardware is detected correctly:

```bash
axdevice
```

Expected output lists detected Metis devices with their firmware versions. If no devices are listed, ensure:
1. You run `axdevice --refresh`
2. The `metis-dkms` driver is loaded (`lsmod | grep metis`)
3. The hardware is seated correctly in its USB or PCIe slot
4. You have sufficient permissions to access the device

> [!TIP]
> When running multiple Metis PCIe cards on the same host, increase the open file descriptor limit to avoid resource exhaustion:
> ```bash
> ulimit -n 10240
> ```


## Slimmed down environments

The recommended way for new users is to install the full SDK environment.
We also provide slimmed down environments that can be used for different purposes.

> [!NOTE]
> The slimmed down environments don't require cloning the `axelera-ai-hub/voyager-sdk.git` repository mentioned above.

### Runtime environment

Install `axelera-rt` on any system that will **only run** compiled models on Metis hardware.

```bash
pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-rt
```

This package provides access to [axrunmodel](/docs/reference/axrunmodel.md) and [axdevice](/docs/reference/axdevice.md) commands.

### Compiler environment

Install `axelera-devkit` on systems where you will **compile** models.

This provides access to the Axelera compiler. For further details, refer to the following documentation:
 - [Compiler CLI](/docs/reference/compiler_cli.md)
 - [Compiler API](/docs/reference/compiler_api.md)
 - [Compiler Config](/docs/reference/compiler_configs_full.md)

> [!WARNING]
> Some documentation pages may refer to [SDK Installer guide](/docs/tutorials/install.md) during a transition period

## Next Steps
- **Firmware updates**: [Firmware Update Guide](/docs/tutorials/firmware_update_decision_tree.md)
- **Deploy a model**: [Pipeline Deployment](/docs/reference/deploy.md)
- **Run first inference**: [Quick Start Guide](/docs/tutorials/quick_start_guide.md)

## Related Documentation
**Tutorials:**
- [Windows Getting Started](/docs/tutorials/windows/windows_getting_started.md) - Windows-specific setup
- [AxMonitor](/docs/tutorials/axmonitor.md) - Monitor and record inference sessions (requires `axelera-voyager-sdk-base`)

**References:**
- [AxDevice API](/docs/reference/axdevice.md) - Verify hardware detection after install
- [AxRuntime API](/docs/reference/axruntime.md) - Low-level Metis device interface
- [inference.py CLI](/docs/reference/inference.md) - Command-line inference and benchmarking tool

## Support

For multi-card PCIe setups on Linux, increase the open file descriptor limit before running workloads:

```bash
ulimit -n 10240
```

## Further support

For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).

For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).
