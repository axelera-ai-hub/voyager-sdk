# Install the Voyager SDK

There are two installation paths. **Python pip** is recommended for new installations.

> [!IMPORTANT]
> **Platform note**
> These instructions are for **Linux** (Ubuntu 22.04+). For Windows, see the [Windows Setup Guide](windows-setup.md) — you will need WSL2 configured first.


| Path | Best for | What it installs |
| :--- | :--- | :--- |
| **[Python pip](#python-pip-installation)** (recommended) | New installations, SDK 1.6+ | `axelera-rt`, `axelera-devkit` via pip |
| **[SDK Installer](#sdk-installer)** | Existing workflows, older SDK versions | Full environment via `install.sh` |

## Prerequisites

- Ubuntu 22.04 or later
- Python 3.10, 3.11, 3.12, or 3.13
- Git installed
- Internet connection
- sudo privileges
- Metis hardware installed and powered on (see [Hardware Installation](../getting-started/hardware-install.md))

---

## Python pip installation

> [!NOTE]
> Always refer to the [Compatibility Matrix](../../RELEASE_COMPATIBILITY_MATRIX.md) for the versions

### Step 1: Clone the repository and install dependencies

```bash
git clone https://github.com/axelera-ai-hub/voyager-sdk.git
cd voyager-sdk
```

Install system dependencies using the provided script:

```bash
./install-dependencies.sh
```
### Step 2: Create a virtual environment and install

```bash
python3 -m venv axelera-env
source axelera-env/bin/activate
pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-rt axelera-devkit[all]
make operators
```

> [!TIP]
> Use a dedicated virtual environment for each SDK version or project to avoid dependency conflicts.

### Step 3: Install the Metis kernel driver

> [!NOTE]
> If installation of the driver fails, ensure your kernel headers are present: `sudo apt-get install -y linux-headers-$(uname -r)`

There are two ways of installing the driver, Option a) is the easiest.

#### Option a) use `axdevice`

```bash
axdevice driver --install
```

By default, `axdevice` will default to installing the recommended driver version 

### Option b) use `apt`

```bash
# Add the Axelera apt repository
sudo sh -c "curl -fsSL https://software.axelera.ai/artifactory/api/security/keypair/axelera/public | gpg --dearmor -o /etc/apt/keyrings/axelera.gpg"
# Ubuntu 22.04
sudo sh -c "echo 'deb [signed-by=/etc/apt/keyrings/axelera.gpg] https://software.axelera.ai/artifactory/axelera-apt-source ubuntu22 main' > /etc/apt/sources.list.d/axelera.list"
# Ubuntu 24.04
sudo sh -c "echo 'deb [signed-by=/etc/apt/keyrings/axelera.gpg] https://software.axelera.ai/artifactory/axelera-apt-source ubuntu24 main' > /etc/apt/sources.list.d/axelera.list"

sudo apt-get update
sudo apt-get install -y metis-dkms
```

Verify the driver is loaded:

```bash
lsmod | grep metis
```

Verify the loaded driver version:
```bash
modinfo metis | grep ^version:
```

Reload the driver:

```bash
sudo modprobe -r metis
sudo modprobe metis
```

### Step 4: Verify installation

```bash
axdevice
```

Expected output lists detected Metis devices with their firmware versions. If no devices are listed, run `axdevice --refresh` and check that the driver is loaded.

### Slimmed-down environments

For deployment-only or compile-only machines:

| Environment | Install command |
| :--- | :--- |
| **Runtime only** | `pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-rt` |
| **Compiler only** | `pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-devkit` |

Proceed to [Verify Your Setup](../getting-started/verify-setup.md).

---

## SDK Installer

> [!WARNING]
> **Deprecation warning.** This installer is deprecated and will be removed in a future release


> [!NOTE]
> The SDK Installer is the previous installation method. For new installations, the [pip path](#python-pip-installation) above is recommended.


### Quickstart

For returning users — the full install sequence:

```bash
git clone https://github.com/axelera-ai-hub/voyager-sdk.git
cd voyager-sdk
./install.sh --all --media
source venv/bin/activate
```

Details for each step below.

### Step 1: Clone the repository

```bash
git clone https://github.com/axelera-ai-hub/voyager-sdk.git
```

This downloads the latest release branch. The repository contains:

| Directory/File | Purpose |
|----------------|---------|
| `install.sh` | Installer |
| `inference.py` | Pipeline evaluation and benchmarking tool |
| `deploy.py` | Pipeline deployment tool |
| `ax_models/` | Models and pipelines |
| `examples/` | Example applications |
| `docs/` | Documentation |

## Step 2: Run the installer

```bash
cd voyager-sdk
./install.sh --all --media
```

The installer sets up:
- Metis PCIe driver (installed to `/lib/modules`)
- Runtime libraries (installed to `/opt/axelera/<version>`)
- Python virtual environment (installed to `~/.cache/axelera/venvs`, symlinked to `./venv`)
- Sample media files for evaluation (the `--media` flag)

> [!NOTE]
> You must re-run the installer each time you checkout a different SDK release branch.


### Installer options

| Option | What it does |
|--------|-------------|
| `--all --media` | Full install with sample videos (recommended for first setup) |
| `--all` | Full install without sample videos |
| `--runtime --no-development` | Runtime only (for deployment machines) |
| `--help` | Show all available options |

### Multiple PCIe cards

If running multiple PCIe cards on the same host, increase the open file descriptor limit:

```bash
ulimit -n 10240
```

## Step 3: Activate the development environment

> [!CAUTION]
> **Every terminal session**
> You must activate the environment in **every new terminal session** before using the SDK. This is the most common setup issue.


```bash
source venv/bin/activate
```

The activation script sets the following environment variables:

| Variable | Description |
|----------|-------------|
| `AXELERA_FRAMEWORK` | Location of the Voyager repository |
| `AXELERA_RUNTIME_DIR` | Host runtime libraries for Metis |
| `AXELERA_DEVICE_DIR` | Metis firmware and low-level binaries |
| `LD_LIBRARY_PATH` | Adds Voyager runtime libraries to the system path |
| `PYTHONPATH` | Adds Voyager SDK to the Python path |

To leave the environment:

```bash
deactivate
```

> [!WARNING]
> **After switching SDK versions**
> If you use `git checkout` to switch to a different release branch, you must deactivate and reactivate the environment. The environment variables are tied to the branch you installed on.
>
> ```bash
> deactivate
> source venv/bin/activate
> ```


## Switching SDK versions

To return to the latest release at any time:

```bash
git fetch --tags
git checkout latest
./install.sh --all
source venv/bin/activate
```

The `latest` tag always points to the current stable release. To switch to a specific older version instead:

```bash
git checkout tags/v1.5.0
./install.sh --all
source venv/bin/activate
```

---

## Next steps

Your SDK is installed and activated. Proceed to [Verify Your Setup](../getting-started/verify-setup.md).
