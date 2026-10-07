---
title: "install.sh"
---
# install.sh — removed

> [!WARNING]
> **`install.sh` has been removed.** Use [`./quick-start.sh`](../../user-guides/sdk-install.md#quick-start) or the `pip` instructions in the [SDK installation guide](../../user-guides/sdk-install.md#python-pip-installation) instead.

`launch-docker.sh` and the `cfg/` configuration files used by the installer were removed together with it.

| Former `install.sh` task | Replacement |
|--------------------------|-------------|
| `--all` (driver, runtime, Python environment) | `./quick-start.sh`, or `pip install axelera-rt axelera-devkit[all]` + `axdevice driver --install` + `make operators` |
| `--runtime --no-development` | `pip install axelera-rt` (see [Slimmed-down environments](../../user-guides/sdk-install.md#slimmed-down-environments)) |
| `--driver` | `axdevice driver --install` |
| `--media` | `axdownloadmedia` |
