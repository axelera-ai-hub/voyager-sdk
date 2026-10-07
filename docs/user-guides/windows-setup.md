---
title: "Windows Setup"
---
# Windows Setup

This guide covers installing the Axelera software on **Windows 11**. The main path is native Windows — you install the driver and Axelera components directly on Windows without WSL.

> [!IMPORTANT]
> **What works where**
>
> | Feature | Ubuntu (Linux) | Windows (Native) | Windows (WSL2) |
> |---------|---------------|------------------|----------------|
> | Network deployment | Yes | No | Yes |
> | Inference | Yes | Pipeline Builder + AxRuntime APIs | No |
>
> Native Windows limitations: no camera capture (file and network stream inputs
> work). Both the Vulkan and OpenCV renderers are available; Vulkan additionally
> needs a GLSL compiler - see
> [Step 4c](#4c-install-a-glsl-compiler-for-the-vulkan-renderer).
>
> For model deployment, you need Linux or WSL2 — see [Step 2](#step-2-optional-install-wsl2-for-model-deployment) below.

---

## Step 1: Install the Windows driver

Windows Update might distribute an older version of the Axelera driver. To check which driver is installed, open a **Command Prompt** or **PowerShell** and run:

```cmd
pnputil /enum-devices /deviceid "PCI\VEN_1F9D" /drivers | findstr /c:"Device Description" /c:"Driver Version"
```

The output lists each Axelera device and its driver:

```text
Device Description:         Metis Device
Driver Version:             09/21/2026 1.4.5.0
```

If every Axelera device reports a **Driver Version** of 1.4.5, complete [1a](#1a-grant-lock-pages-in-memory-privilege) and continue with [Step 2](#step-2-optional-install-wsl2-for-model-deployment). Otherwise, or if no driver is listed, follow the steps below to install it manually.

With a 150p card, install the Switchtec driver manually in any case, as described in [1e](#1e-install-the-switchtec-driver-150p-card-only).

### 1a. Grant "Lock pages in memory" privilege

This step is required once per machine. The Axelera runtime needs the **Lock pages in memory** privilege to operate correctly.

1. Press **Win + S**, type `secpol.msc`, and press **Enter** to open the Local Security Policy editor
2. In the left pane, navigate to **Local Policies → User Rights Assignment**
3. In the right pane, double-click **Lock pages in memory**
4. Click **Add User or Group**, enter your Windows username, and confirm

![Lock Pages in Memory](../images/windows/enable_lock_memory.png)

Log out and back in (or reboot) for the privilege to take effect.

### 1b. Download the drivers

Download the driver archives from the Axelera software portal:

- **Metis driver** (required for all hardware): [Metis-1.4.5.zip](https://software.axelera.ai/artifactory/axelera-win/driver/1.4.x/Metis-1.4.5.zip)
- **Switchtec PCIe switch driver** (required only for the [Axelera Server 150p (PCIe 4-AIPU) card](https://store.axelera.ai/collections/ai-acceleration-cards/products/pcie-ai-accelerator-card-powered-by-4-metis-aipu)): [switchtec-kmdf-0.7_2019.zip](https://software.axelera.ai/artifactory/axelera-win/driver/1.3.x/switchtec-kmdf-0.7_2019.zip)

Extract each archive locally. Each extracted folder contains three files: `.cat`, `.inf`, and `.sys`.

### 1c. Remove any previous driver version

> [!TIP]
> **First-time install?** Skip this step and go straight to [1d](#1d-install-via-device-manager).

If you already have an older Metis driver installed, remove it completely before installing the new one. Windows can keep multiple drivers for the same device.

In Device Manager, locate the Metis device under **System devices**, right-click and select **Uninstall device**:

![Uninstall Device](../images/windows/uninstall_device.png)

In the confirmation dialog, check **"Attempt to remove the driver for this device"**, then click **Uninstall**:

![Remove Driver Checkbox](../images/windows/uninstall_remove_driver.png)

Verify the device now appears under **Other devices** as a generic **PCI Device** — this confirms the driver is fully removed:

![PCI Device](../images/windows/pci_device_other_devices.png)

### 1d. Install via Device Manager

Open **Device Manager** (right-click the Windows Start button → Device Manager).

From the menu, open **Action → Add Drivers**:

![Add Driver](../images/windows/add_driver.png)

Set the path to the folder where you extracted the Metis driver files:

![Set Path](../images/windows/set_path.png)

Confirm that the Metis device now appears under **System devices**:

![Metis Device](../images/windows/metis_device.png)

### 1e. Install the Switchtec driver (150p card only)

Repeat the steps in [1d](#1d-install-via-device-manager), this time setting the path to the folder where you extracted the Switchtec driver files. Confirm that the Switchtec device appears:

![Switchtec Device](../images/windows/switchtec_device.png)

---

## Step 2 (Optional): Install WSL2 for model deployment

> [!NOTE]
> **Skip this step if you only need inference.** If you want to deploy your own models or use the full SDK tooling, install WSL2 and follow the Linux setup inside it.

Network deployment is supported exclusively in Linux. WSL2 is the recommended approach for deploying models for Windows use.

### 2a. Install WSL2

Open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu-22.04
```

Restart your computer when prompted. After restart, Ubuntu opens and asks you to create a username and password.

> [!TIP]
> **Already have WSL?** Check your version: `wsl --list --verbose`. You need WSL **2** (not 1). Upgrade with `wsl --set-version Ubuntu-22.04 2`.

### 2b. Clone the SDK in a Windows-accessible folder

After the restart, open a regular **Command Prompt** (not PowerShell, not Administrator) and enter WSL:

```cmd
wsl
```

Inside WSL, clone to a path under `/mnt/c/` so both Windows and WSL can reach it:

```bash
# /mnt/c/ maps to C:\ on Windows
cd /mnt/c/Axelera
git clone https://github.com/axelera-ai-hub/voyager-sdk.git
cd voyager-sdk
```

This means:
- From **WSL**: `/mnt/c/Axelera/voyager-sdk`
- From **Windows**: `C:\Axelera\voyager-sdk`

### 2c. Follow the Linux setup inside WSL

Install the SDK inside WSL, then use it for model deployment:

1. [SDK Install](sdk-install.md) — clone, install, and activate
2. [deploy.py](../reference/tools/deploy-py.md) — compile and deploy models for use on Windows

> [!WARNING]
> When installing the SDK inside WSL with `pip`, **do not** install the driver (`axdevice driver --install`) — the driver was installed on the Windows side in Step 1, not inside WSL.
>
> ![WSL Installation Options](../images/windows/wsl_voyager_install.png)

> [!NOTE]
> The Metis device is not accessible from inside WSL. Hardware verification and inference run on the Windows side (Steps 3–6 below).

---

## Step 3: Install Python (Windows native)

> [!NOTE]
> Perform this step in Windows natively, **not in WSL**.

1. Download [Python for Windows](https://www.python.org/downloads/windows/) from the official website. The latest stable release tested by Axelera is 3.13.5.

> [!IMPORTANT]
> Install Python from the official website link above — **not** from the Microsoft Store.

2. During installation:
   - Enable **"Add Python to PATH"**
   - Select **"Disable path length limit"** when prompted

3. Add the Python installation path and the `Python\Scripts` path to the Windows **Path** environment variable:

   ![Python Environment Variables](../images/windows/system_variable_python.png)

4. Create a virtual environment:

   ```cmd
   python -m venv venv-win
   ```

---

## Step 4: Install Windows Axelera components

> [!NOTE]
> Perform this step in Windows natively, **not in WSL**.

Open a regular **Command Prompt** (not PowerShell, not Administrator) and navigate to your SDK directory:

```cmd
cd C:\Axelera\voyager-sdk
```

Create a local download directory and download all packages:

```cmd
rmdir /s /q windows-packages 2>nul
mkdir windows-packages
cd windows-packages

curl -L -O https://software.axelera.ai/artifactory/axelera-win/packages/1.9.x/axelera-win-device-installer.exe
curl -L -O https://software.axelera.ai/artifactory/axelera-win/packages/1.9.x/axelera-win-runtime-installer.exe
curl -L -O https://software.axelera.ai/artifactory/axelera-win/packages/1.9.x/axelera-win-services-installer.exe

cd ..
```

### 4a. Install the executable packages

> [!IMPORTANT]
> If any previous Axelera packages are installed, uninstall them via **Apps & Features** before proceeding.

Using **Windows Explorer**, navigate to `C:\Axelera\voyager-sdk\windows-packages` and run each `.exe` installer in sequence.

After installation, verify via **Settings → Apps & features** (or **Add or remove programs**):

![Add Remove Programs](../images/windows/add_remove_choice.png)

All three packages must be listed:

![Axelera Packages](../images/windows/add_remove_axelera_packages.png)

You must see: **Axelera Device Package**, **Axelera Runtime**, and **Axelera Services**. Exact version numbers will vary.

**Close all Command Prompt and PowerShell windows** after the installers complete, then open a new Command Prompt before continuing.

### 4b. Install Python packages

Activate the virtual environment and install the packages:

```cmd
venv-win\Scripts\activate.bat

pip install --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple axelera-runtime==1.9.0 axelera-llm==1.9.0 axelera-types==1.9.0 axelera-runtime2
```

`axelera-runtime2` provides the Pipeline Builder API; install the version
matching your SDK release. It requires the Microsoft Visual C++
Redistributable (x64), which most systems already have.

> [!TIP]
> **Re-activating the environment:** Remember to activate the virtual environment each time you open a new Command Prompt. From your SDK directory:
> ```cmd
> venv-win\Scripts\activate.bat
> ```
> You'll know it's active when `(venv-win)` appears at the start of your prompt.

### 4c. Install a GLSL compiler (for the Vulkan renderer)

The Vulkan renderer compiles its shaders at runtime and needs a GLSL compiler.
Skip this step if you only use `--display opencv`, `console` or `none`; on Linux
the equivalent is the `glslang-tools` package.

**Recommended - the Vulkan SDK via winget:**

```cmd
winget install KhronosGroup.VulkanSDK
```

The SDK sets `VULKAN_SDK`, which the renderer searches directly, so no **Path**
change is needed. winget tracks upstream releases, so there is no version to
keep up to date here.

**Alternative - glslang on its own.** The SDK is a large install for a single
executable. To fetch just the compiler, in PowerShell:

```powershell
$asset = (Invoke-RestMethod https://api.github.com/repos/KhronosGroup/glslang/releases/latest).assets |
    Where-Object name -like '*windows-x86_64-release.zip'
Invoke-WebRequest $asset.browser_download_url -OutFile "$env:TEMP\glslang.zip"
Expand-Archive "$env:TEMP\glslang.zip" -DestinationPath C:\glslang -Force
```

Then add `C:\glslang\bin` to the Windows **Path** environment variable.

**Verify.** Open a new Command Prompt so the environment change is picked up,
activate the virtual environment, and run:

```cmd
python -c "from axelera.runtime2 import display_vk_shaders as s; print(s.compiler())"
```

This performs the same lookup as the renderer and prints the compiler it found.

---

## Step 5: Run examples

First, obtain a deployed model. If you completed the WSL2 setup in Step 2 and deployed a model there, it is accessible from Windows at `C:\Axelera\voyager-sdk\build\`. Otherwise, download a pre-built model:

```cmd
axdownloadmodel resnet50-imagenet-onnx
```

The model downloads to `C:\Axelera\voyager-sdk\build\`.

**Run inference using the `axrunmodel` utility** (measures performance):

```cmd
axrunmodel C:\Axelera\voyager-sdk\build\resnet50-imagenet-onnx\resnet50-imagenet-onnx\1\model.json
```

**Run inference using the Python API:**

Download an image to `C:\Axelera\voyager-sdk\examples\axruntime\images`, then:

```cmd
cd examples\axruntime
python -m axruntime_example -v --aipu-cores 1 --labels imagenet-labels.txt ..\..\build\resnet50-imagenet-onnx\resnet50-imagenet-onnx\1\model.json images
```

### Run a model recipe with the Pipeline Builder API

Model recipes under `model_recipes/` run natively on Windows against a
pre-deployed model. Deployment itself stays in Linux/WSL2: from WSL (Step 2),
build the model once -

```bash
python model_recipes/pose/yolo11n_pose_coco_axrelay.py --build
```

- which writes `build/yolo11n-pose-coco.axm` into the shared checkout.

Then, back in native Windows with the virtual environment active, run the same
recipe against that artifact:

```cmd
python model_recipes\pose\yolo11n_pose_coco_axrelay.py --axm build\yolo11n-pose-coco.axm --run image.jpg --display none
```

Detections are printed to the console. To render them in a window instead, use
`--display vulkan` (the preferred renderer, requiring the GLSL compiler from
[Step 4c](#4c-install-a-glsl-compiler-for-the-vulkan-renderer)), `--display opencv`, or
`--display auto` to pick the best available. Inputs can be image files, video
files, or network streams (RTSP/HTTP) - cameras are not supported on Windows.

---

## Step 6: Run LLM models

A set of precompiled LLM models can run natively on Windows. To see available models:

```cmd
axllm --help-network
```

Run a model in single-prompt mode:

```cmd
axllm llama-3-2-1b-1024-4core-static --prompt "Give me a joke"
```

Run interactively in chat mode:

```cmd
axllm llama-3-2-1b-1024-4core-static
```

For all options, see the [LLM Inference Guide](llm.md).

---

## Using PowerShell

After completing installation, you can use PowerShell instead of Command Prompt. The only difference is how you activate the virtual environment:

```powershell
.\venv-win\Scripts\Activate.ps1
```

Once activated, all the same commands work as in Command Prompt.

> [!NOTE]
> **PowerShell execution policy**
> If you get an error activating the environment in PowerShell, your execution policy may be blocking unsigned scripts. Check your current policy:
> ```powershell
> Get-ExecutionPolicy
> ```
> If needed, set a more permissive policy (run PowerShell as Administrator):
> ```powershell
> Set-ExecutionPolicy RemoteSigned
> ```
>
> Use **Command Prompt** (not PowerShell) for the installation steps — particularly Step 4a. PowerShell is fine for running examples and day-to-day use after setup is complete.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Metis not detected in Device Manager | Confirm the driver installed correctly — see [Step 1d](#1d-install-via-device-manager) |
| `wsl --install` fails | Enable "Virtual Machine Platform" in Windows Features |
| WSL shows version 1 | `wsl --set-version Ubuntu-22.04 2` |
| PowerShell can't activate venv | Run `Set-ExecutionPolicy RemoteSigned` as Administrator, then retry |
| Installer fails silently | Uninstall all previous Axelera packages first via Apps & Features |
| `WinError 126` loading `libaxruntime.dll` | The installers set user environment variables that existing terminals do not see — close every Command Prompt/PowerShell window and open a new one |
| `no AIPU cores available to run on: none were detected` | The same stale-environment cause as `WinError 126`: `AIPU_FIRMWARE_OMEGA` and `AIPU_RUNTIME_STAGE0_OMEGA` still point at an SDK version that is not installed, so the firmware cannot be read. Open a new terminal, or sign out and back in |
| Environment still stale in a newly opened terminal | Reconnecting a *disconnected* Remote Desktop session restores that session's original process tree, so even a shell launched from the Start menu inherits the environment captured at the original logon. Sign out and back in - reconnecting is not enough |
| `axdevice` works but inference fails with `zeInit` `UNINITIALIZED` at device connect | Reboot after installing the three packages in Step 4a |
| `--display vulkan` reports that the shader compiler was not found | Install a GLSL compiler ([Step 4c](#4c-install-a-glsl-compiler-for-the-vulkan-renderer)), then open a new Command Prompt so the environment change is picked up |
| `--display vulkan` opens a window that stays blank | The session has no physical monitor. A headless session reports a synthesized 640x480 display rather than none, so the renderer cannot tell it apart from a real one. Attach a monitor, or connect over Remote Desktop, which supplies a display adapter |
| The renderer falls back to the console, reporting that no monitor is attached | The session has no presentation target - a disconnected console session with no monitor and no indirect-display driver. Attach a monitor, or connect over Remote Desktop, which supplies a display adapter |
| `--display vulkan` renders nothing over Remote Desktop | Enable the "Use hardware graphics adapters for all Remote Desktop Services sessions" group policy, or run at the physical console |

---

## See also

- [SDK Install](sdk-install.md) — the main Linux/WSL installation guide
- [Verify Setup](../getting-started/verify-setup.md) — confirm hardware and SDK are working
- [LLM Inference Guide](llm.md) — full LLM options including Windows
