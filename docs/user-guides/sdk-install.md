# Install the Voyager SDK

> [!IMPORTANT]
> **Platform note**
> These instructions are for **Linux** (Ubuntu 22.04+). For Windows, see the [Windows Setup Guide](windows-setup.md) — you will need WSL2 configured first.

## Quick start
For quick starting your Voyager SDK installation, run:
```bash
git clone https://github.com/axelera-ai-hub/voyager-sdk.git
cd voyager-sdk
./quick-start.sh
```
The script will install all required dependencies and create a python virtual environment with the full SDK dependencies.

It installs the SDK version recorded in `VERSION.TXT`.

If you want more control, continue reading.

## In-depth guide
> [!WARNING]
> **The `install.sh` SDK installer has been removed.** Use `./quick-start.sh` (above) or the `pip` instructions below instead.

The supported installation path is **Python pip**.

| Path | Best for | What it installs |
| :--- | :--- | :--- |
| **[Python pip](#python-pip-installation)** | New installations, SDK 1.6+ | `axelera-rt`, `axelera-devkit` via pip |

### Prerequisites

- Ubuntu 22.04 or later
- Python 3.10, 3.11, 3.12, or 3.13
- Git installed
- Internet connection
- sudo privileges
- Metis hardware installed and powered on (see [Hardware Installation](../getting-started/hardware-install.md))

---

### Python pip installation

> [!NOTE]
> Always refer to the [Compatibility Matrix](../../RELEASE_COMPATIBILITY_MATRIX.md) for the versions

#### Step 1: Clone the repository and install dependencies

```bash
git clone https://github.com/axelera-ai-hub/voyager-sdk.git
cd voyager-sdk
```

Install system dependencies using the provided script:

```bash
./install-dependencies.sh
```
#### Step 2: Create a virtual environment and install

```bash
python3 -m venv axelera-env
source axelera-env/bin/activate
pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-rt axelera-devkit[all]
make operators
```

> [!TIP]
> Use a dedicated virtual environment for each SDK version or project to avoid dependency conflicts.

#### Step 3: Install the Metis kernel driver

> [!NOTE]
> If installation of the driver fails, ensure your kernel headers are present: `sudo apt-get install -y linux-headers-$(uname -r)`

There are two ways of installing the driver, Option a) is the easiest.

##### Option a) use `axdevice`

```bash
axdevice driver --install
```

By default, `axdevice` will default to installing the recommended driver version 

##### Option b) use `apt`

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

#### Step 4: Verify installation

```bash
axdevice
```

Expected output lists detected Metis devices with their firmware versions. If no devices are listed, run `axdevice --refresh` and check that the driver is loaded.

### Slimmed-down environments

For deployment-only or compile-only machines:

| Environment | Install command |
| :--- | :--- |
| **Runtime only - [axrunmodel](../reference/tools/axrunmodel.md)** | `pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-rt` |
| **Runtime only - [inference.py](../reference/tools/inference-py.md)** | `pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-rt[all]` |
| **Compiler only - [deploy.py](../reference/tools/deploy-py.md)** | `pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-devkit[all]` |

> [!NOTE]
> The `[all]` extra is what makes `deploy.py` and `inference.py` work. Plain
> `axelera-rt` and `axelera-devkit` carry the runtime and the compiler, but not
> the Python libraries `axelera.app` imports on top of them, such as
> `strictyaml` and `opencv-python`. Install without the extra only if you are
> calling the compiler or runtime APIs directly.

Proceed to [Verify Your Setup](../getting-started/verify-setup.md).

---

### SDK Installer (removed)

> [!WARNING]
> **`install.sh` has been removed** (along with `launch-docker.sh`, `installer_support.py` and the `cfg/` files it used). Use [`./quick-start.sh`](#quick-start) or the [Python pip installation](#python-pip-installation) above.
>
> - Virtual environments are now created by `quick-start.sh` as `./venv_<version>`; activate with `source venv_<version>/bin/activate`.
> - To switch SDK versions, `git checkout` the desired tag and run `./quick-start.sh` again.
> - Sample videos (formerly `--media`) are no longer downloaded by an installer; use `axdownloadmedia` to list and download them.

---

### Next steps

Your SDK is installed and activated. Proceed to [Verify Your Setup](../getting-started/verify-setup.md).
