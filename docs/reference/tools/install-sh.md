---
title: "install.sh"
---
# install.sh — SDK Installer

The `install.sh` script inspects your system and installs everything needed to run AI models on Axelera hardware. You run it once after cloning the SDK, and again each time you switch to a different SDK release.

## Quick reference

```bash
# Full install with sample videos
./install.sh --all --media

# Full install without sample videos
./install.sh --all

# Runtime only (no development tools)
./install.sh --runtime --no-development

# See all options
./install.sh --help
```

## What it installs

The installer sets up three components:

| Component | Location | Purpose |
|-----------|----------|---------|
| PCIe driver | `/lib/modules/` | Kernel module for Metis hardware communication |
| Runtime libraries | `/opt/axelera/<version>/` | Host-side libraries for device access and model execution |
| Python virtual environment | `~/.cache/axelera/venvs/` (symlinked to `./venv`) | Python packages for the complete SDK toolchain |

### PCIe driver

Installed to the system location. Contains minimal functionality that rarely changes between SDK releases. Requires `sudo` privileges.

### Runtime libraries

Installed to `/opt/axelera/` in version-specific directories. This means you can have multiple SDK versions installed side by side — each with its own runtime libraries.

### Python virtual environment

Installed to `~/.cache/axelera/venvs/` and symlinked to `./venv` in the SDK directory. Contains all Python dependencies for compilation, inference, and deployment.

> [!CAUTION]
> **Always activate the environment**
> After installation, you must activate the virtual environment before using any SDK tool:
>
> ```bash
> source venv/bin/activate
> ```
>
> Without this, commands like `inference.py` and `axdevice` will not work.


## Common options

| Option | Purpose |
|--------|---------|
| `--all` | Install all components (driver, runtime, Python environment) |
| `--media` | Download sample videos for testing (traffic, pedestrians, etc.) |
| `--runtime --no-development` | Install only what's needed to run models (skip compiler/development tools) |
| `--help` | Show all available options |

## When to re-run

You must re-run `install.sh` when:

1. **Switching SDK versions** — Each release may have different dependencies
2. **After a fresh clone** — The virtual environment doesn't exist yet
3. **After system updates** — Kernel updates may require driver reinstallation

You do **not** need to re-run when:
- Running inference with an already-installed SDK version
- Changing models or pipelines
- Updating your own application code

## Repository structure

After cloning, the SDK repository contains:

| Path | What it is |
|------|-----------|
| `install.sh` | This installer |
| `cfg/` | Configuration files per platform (e.g., `config-ubuntu-2204-amd64.yaml`) |
| `inference.py` | [Pipeline evaluation tool](inference-py.md) |
| `deploy.py` | Pipeline deployment tool |
| `ax_models/` | [Model Zoo](../models/model-zoo.md) — pre-trained models and pipelines |
| `examples/` | Example applications |
| `docs/` | Documentation |

## Multiple SDK versions

You can install multiple SDK versions on the same system:

```bash
# Clone and install v1.5
git clone https://github.com/axelera-ai-hub/voyager-sdk.git sdk-v1.5
cd sdk-v1.5
./install.sh --all

# Clone and install v1.4
cd ..
git clone https://github.com/axelera-ai-hub/voyager-sdk.git sdk-v1.4
cd sdk-v1.4
git checkout release/v1.4.0
./install.sh --all
```

Each version has its own `venv/` and runtime libraries. Switch between them by activating the appropriate environment.

> [!IMPORTANT]
> When switching SDK versions, always `deactivate` the current environment first, then `source venv/bin/activate` in the new SDK directory.


## See also

- [SDK Install](../../user-guides/sdk-install.md) — step-by-step installation walkthrough
- [Virtual Environments](../system/virtual-environments.md) — why activation matters
- [Environment Variables](../system/environment-variables.md) — what `source venv/bin/activate` sets
