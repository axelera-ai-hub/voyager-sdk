![](/docs/images/Ax_Page_Banner_2500x168_01.png)

# Installation guide

## Contents
- [Installation guide](#installation-guide)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Level](#level)
  - [Overview](#overview)
  - [Run the installer](#run-the-installer)
  - [Activate the development environment](#activate-the-development-environment)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Support](#support)
  - [Further support](#further-support)

## Prerequisites
- Ubuntu 22.04+ or Windows with WSL2 installed
- Python 3.10 or later
- Git installed
- Internet connection for downloading dependencies
- Administrative/sudo privileges
- USB 3.0 port or PCIe slot for Metis hardware

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
The repository contains the following files and directories.

| File or directory | Description |
| :---------------- | :---------- |
| [`install.sh`](/install.sh) | Installer tool |
| [`cfg/`](/cfg) | Installer configuration files |
| [`deploy.py`](/deploy.py) | [Pipeline deployment tool](/docs/reference/deploy.md) |
| [`inference.py`](/inference.py) | [Pipeline evaluation and benchmarking tool](/docs/reference/inference.md) |
| [`ax_models`](/ax_models) | [Models and pipelines](/docs/reference/model_zoo.md) |
| [`examples/`](/examples) | Example applications utilizing different models and pipelines |
| [`docs`](/docs/) | Tutorial and reference documentation |
| [`licenses/`](/licenses) | Licenses for all SDK components and dependencies |

The default branch of the repository is always set to the latest published SDK release. You can use
standard `git` commands to list the available releases and to checkout different versions of the
SDK. Run the following command to  view the current release branch and all available releases:

```bash
git branch
```

To checkout a specific SDK release, run a command such as:

```
git checkout release/v1.2.5
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

## Run the installer

You must run the installer each time you checkout a new SDK release.
The following command installs everything you need to deploy and run models on Metis hardware:

```bash
./install.sh --all --media
```

The installer inspects the system and ensures the Metis PCIe driver, system runtime
libraries and Python virtual environment are all correctly installed. The Python virtual
environment contains all libraries needed by the complete toolchain. All
of these components and their version numbers are listed in the installer configuration file for your specific host,
for example [`cfg/config-ubuntu-2204-amd64.yaml`](/cfg/config-ubuntu-2204-amd64.yaml).

The `--media` option downloads a collection of sample videos which you can use to evaluate
different models visually.

The PCIe driver is installed to the system location `/lib/modules`. It contains minimal
functionality that does not generally change between releases.

Most runtime functionality is provided in libraries installed to
`/opt/axelera/`<*version specific directories*>. This enables you to run multiple applications
built with different SDK versions on the same system.

The Python environments are installed to `~/.cache/axelera/venvs` (and symlinked to `./venv`).
You can install multiple development environments on the system and switch between them easily.

If you plan to deploy models on one development machine and then run them on different hosts,
run the installer on each system with no options and it will prompt you to select which
components you require. Running the installer with the `--help` option gives more
information on how to enable and disable these individual components on the command line.

> [!TIP]
> To install only the runtime on a deployment system run the installer
> with the options `--runtime` and `--no-development`.

## Activate the development environment

To activate the installed development environment, run the following command:

```bash
source venv/bin/activate
```

The activation script sets a number of environment variables based on the current SDK release
(as specified in the installer configuration file).

| **Environment variable** | **Default value** | **Description** |
| -------------------------| ------------------| --------------- |
| `AXELERA_FRAMEWORK` | `.` | Location of the Voyager repository |
| `AXELERA_DEVICE_DIR` | `/opt/axelera/device-<version>` | Metis firmware and low-level binaries |
| `AXELERA_RUNTIME_DIR` | `/opt/axelera/runtime-<version>/omega` | Host runtime libraries for Metis devices |
| `AXELERA_RISCV_TOOLCHAIN_DIR` | `/opt/axelera/riscv-gnu-newlib-toolchain-<version>` | RISC-V toolchain for Metis on-device control (used by the host runtime) |
| `PYTHONPATH` | `$AXELERA_RUNTIME_DIR/tvm/tvm-src` | Adds Voyager SDK repository to the Python library path |
| `LD_LIBRARY_PATH` | `$AXELERA_RUNTIME_DIR/lib:`  <br>`$AXELERA_FRAMEWORK/operators/lib` | Adds Voyager runtime libraries to the system dynamic shared library path |

When you've finished working with the development environment, type `deactivate` to leave the environment.
All environment variables that were previously set during activation are restored to their original values.

Each time you use git to change the SDK release branch, you should deactivate and then reactivate your
environment. This ensures that your environment variables are correctly set to the values in the
installer configuration file on the new release branch.

## Next Steps
- **Verify installation**: Run `axdevice` to confirm hardware is detected
- **Run first inference**: [Quick Start Guide](quick_start_guide.md)
- **Enable firmware updates**: [Enable Updates Guide](enable_updates.md) for new boards
- **For Windows users**: Complete [Windows Getting Started](windows/windows_getting_started.md)

## Related Documentation
**Tutorials:**
- [Quick Start Guide](quick_start_guide.md) - Run your first inference after installation
- [Windows Getting Started](windows/windows_getting_started.md) - Windows-specific setup
- [Installing Driver](windows/installing_driver.md) - Windows driver installation
- [Enable Updates](enable_updates.md) - Enable firmware updates (one-time per board)

**References:**
- [AxDevice API](../reference/axdevice.md) - Verify hardware detection after install

## Support

In case of using multiple PCIe cards on the same host system it is recommended to increase the number of open file descriptors when running Linux OS:
`ulimit -n 10240`

## Further support
- For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).
- For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).
