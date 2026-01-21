![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Voyager SDK Installation Guide for Windows

## Contents
- [Voyager SDK Installation Guide for Windows](#voyager-sdk-installation-guide-for-windows)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Level](#level)
  - [IMPORTANT NOTICE](#important-notice)
  - [Supported Features Comparison](#supported-features-comparison)
  - [Contents](#contents-1)
  - [Overview](#overview)
  - [Step 1: Install the Axelera Windows Driver](#step-1-install-the-axelera-windows-driver)
  - [Step 2: Install the Axelera Voyager SDK using Windows Subsystem for Linux](#step-2-install-the-axelera-voyager-sdk-using-windows-subsystem-for-linux)
    - [Setting up the voyager-sdk repository on Windows:](#setting-up-the-voyager-sdk-repository-on-windows)
  - [Step 3: Install Python](#step-3-install-python)
  - [Step 4: Install Windows Axelera Components](#step-4-install-windows-axelera-components)
    - [Installing the Executable Installers](#installing-the-executable-installers)
    - [Installing the Python Packages](#installing-the-python-packages)
  - [Step 5: Running Examples \[Optional\]](#step-5-running-examples-optional)
  - [Step 6: Running LLM Models](#step-6-running-llm-models)
  - [Notes on Using Windows PowerShell](#notes-on-using-windows-powershell)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Further support](#further-support)


## Prerequisites
- Windows 11 (only tested version)
- Windows Subsystem for Linux (WSL2) support
- Administrative privileges
- Internet connection

## Level
**Beginner** - Follow step-by-step Windows-specific installation

## IMPORTANT NOTICE

> **Important:** Full development environment is **not available natively on Windows**.
> - **Recommended:** Use **Ubuntu** or **Windows Subsystem for Linux (WSL)** for development.  
> - **Windows SDK** currently supports **inference only** (via AxRuntime API).  
> For details, see Supported Features on Windows vs. Linux.

## Supported Features Comparison

| Feature                | Ubuntu (Linux) | Windows (Native) | Windows (WSL) |
|------------------------|---------------|-------------------|---------------|
| Full SDK Development   | ✅            | ❌                | ✅            |
| Network Deployment     | ✅            | ❌                | ✅            |
| Inference (AxRuntime)  | ✅            | ✅                | ✅            |

## Contents

- [Voyager SDK Installation Guide for Windows](#voyager-sdk-installation-guide-for-windows)
  - [Overview](#overview)
  - [Step 1: Install the Axelera Windows Driver](#step-1-install-the-axelera-windows-driver)
  - [Step 2: Install the Axelera Voyager SDK using Windows Subsystem for Linux](#step-2-install-the-axelera-voyager-sdk-using-windows-subsystem-for-linux)
    - [Setting up the voyager-sdk repository on Windows:](#setting-up-the-voyager-sdk-repository-on-windows)
  - [Step 3: Install Python](#step-3-install-python)
  - [Step 4: Install Windows Axelera Components](#step-4-install-windows-axelera-components)
    - [Installing the Executable Installers](#installing-the-executable-installers)
    - [Installing the Python Packages](#installing-the-python-packages)
  - [Step 5: Running Examples \[Optional\]](#step-5-running-examples-optional)
  - [Step 6: Running LLM Models](#step-6-running-llm-models)
  - [Notes on Using Windows PowerShell](#notes-on-using-windows-powershell)

## Overview
This guide provides step-by-step instructions for installing and configuring the Voyager SDK on a **Windows 11** system.
> [!NOTE]  
> Only Windows 11 with Windows Subsystem for Linux (WSL) support was tested.

Before proceeding, please follow the [Voyager SDK Repository on Windows](/docs/tutorials/windows/windows_voyager_sdk_repository.md) guide to clone the Voyager SDK repository and set up the environment.

## Step 1: Install the Axelera Windows Driver
> [!NOTE]  
> This step will be automated through Windows Update after Microsoft certification in the near future.

For evaluation purposes, follow the detailed instructions in the [Windows Driver Installation Guide](/docs/tutorials/windows/installing_driver.md) for manual installation of the Axelera AI driver on your Windows system.

## Step 2: Install the Axelera Voyager SDK using Windows Subsystem for Linux
> [!NOTE]  
> If you're only interested in downloading pre-deployed networks instead of deploying your own, 
> then you can skip this step and proceed directly to [Step 3: Install Python](#step-3-install-python).

Network deployment is supported exclusively in Linux, even when inference is performed on Windows. Windows Subsystem for Linux (WSL) is the recommended approach for deploying networks for Windows use.

### Setting up the voyager-sdk repository on Windows:

a) **Install Windows Subsystem for Linux (WSL)**:
   - Launch PowerShell as Administrator and execute:
     ```powershell
     wsl --install
     ```
   - Restart the computer after installation completes.
   - To install Ubuntu 22.04 (recommended), execute:
     ```powershell
     wsl --install -d Ubuntu-22.04
     ```
   - Follow the prompts to set up the Ubuntu username and password.
   
   For comprehensive instructions, refer to:
   - [WSL Installation Guide](https://learn.microsoft.com/en-us/windows/wsl/install)
   - [Changing Default Linux Distribution](https://learn.microsoft.com/en-us/windows/wsl/install#change-the-default-linux-distribution-installed)

> [!IMPORTANT]  
> After completing the WSL installation, close the Administrator PowerShell window, open a regular Windows **Command Prompt** (not Windows PowerShell!) without administrator privileges, and launch WSL by entering:
```cmd
wsl
```

b) **Clone and Configure voyager-sdk**:
   - Clone the voyager-sdk repository from https://github.com/axelera-ai-hub/voyager-sdk and checkout branch release/v1.4 **in a Windows-accessible folder**. For example: `/mnt/c/Axelera/voyager-sdk` (accessible from Windows as `C:\Axelera\voyager-sdk`). Complete cloning instructions are available in the [Installation guide](/docs/tutorials/install.md).
   - Follow the [Installation guide](/docs/tutorials/install.md) setup instructions, selecting only the Python environment and runtime libraries during installation
   - **Important:** Do not select the driver installation option as it is not compatible with WSL
   ![WSL Installation Options](/docs/images/windows/wsl_voyager_install.png)

c) **Optional: Deploy or Download Models**:
   While in WSL, activate the Python virtual environment:
   ```bash
   source venv/bin/activate
   ```
   Two options exist for obtaining models:
   1. Deploy models using the Linux environment:
      ```bash
      ./deploy.py [options]
      ```
   2. Download prebuilt models after activating the Python environment:
      ```bash
      source venv/bin/activate
      axdownloadmodel
      ```
   Example:
   ```bash
   axdownloadmodel resnet50-imagenet-onnx
   ```
   This downloads the ResNet50 classification model for future use.
   > [!NOTE]  
   > Additional models can be deployed or downloaded at any time

## Step 3: Install Python
> [!NOTE]  
> Perform this step in Windows natively, **not in WSL**.

1. Download [Python for Windows](https://www.python.org/downloads/windows/). The latest Python Stable Release version is recommended. The latest stable release tested by Axelera is 3.13.5.
> [!IMPORTANT]  
> Install Python from the official website link above, not from the Microsoft Store
2. During installation:
   - Enable the option to add Python to PATH
   - Select "Disable path length limit" when prompted
3. Create a virtual environment:
   ```cmd
   python -m venv venv
   ```
4. Add the Python installation path and Python\Scripts path to the Windows environment Path variable:
   ![Python Environment Variables](/docs/images/windows/system_variable_python.png)

## Step 4: Install Windows Axelera Components
> [!NOTE]  
> Perform this step in Windows natively, **not in WSL**.

Open a Windows regular **Command Prompt** (not Windows PowerShell!) without administrator privileges and navigate to the Voyager SDK installation directory. For example, if installed in `C:\Axelera\voyager-sdk`:
```cmd
cd C:\Axelera\voyager-sdk
```
Create a local windows-packages directory and download the installers:
```cmd
rmdir /s /q windows-packages 2>nul
mkdir windows-packages
cd windows-packages
curl -O https://media.axelera.ai/releases/v1.5.0/build-Release-Windows-2204-amd64/package_repos/axelera-win-device-installer.exe
curl -O https://media.axelera.ai/releases/v1.5.0/build-Release-Windows-2204-amd64/package_repos/axelera-win-runtime-installer.exe
curl -O https://media.axelera.ai/releases/v1.5.0/build-Release-Windows-2204-amd64/package_repos/axelera-win-syslibs-installer.exe
curl -O https://media.axelera.ai/releases/v1.5.0/build-Release-Windows-2204-amd64/package_repos/axelera-win-toolchain-deps-installer.exe
curl -O https://media.axelera.ai/releases/v1.5.0/build-Release-Windows-2204-amd64/package_repos/axelera-win-services-installer.exe
curl -O https://media.axelera.ai/releases/v1.5.0/build-Release-Windows-2204-amd64/package_repos/axelera_llm-1.5.0-py3-none-any.whl
curl -O https://media.axelera.ai/releases/v1.5.0/build-Release-Windows-2204-amd64/package_repos/axelera_runtime-1.5.0-py3-none-any.whl
curl -O https://media.axelera.ai/releases/v1.5.0/build-Release-Windows-2204-amd64/package_repos/axelera_types-1.5.0-py3-none-any.whl
cd ..
```

### Installing the Executable Installers
> [!IMPORTANT]  
> Before installing the executable installers, uninstall any previous versions of Axelera packages if they exist. After installing all executable installers, close all Command Prompt and Windows PowerShell windows. Open a new Command Prompt window (without administrator privileges) to continue with the remaining steps.

Using Windows Explorer, navigate to `C:\Axelera\voyager-sdk\windows-packages` and install each executable installer sequentially.

After installation, verify the packages were installed correctly:

1. Open Windows Settings and navigate to "Apps & features" (or "Add or remove programs")
   
![Add Remove Programs](/docs/images/windows/add_remove_choice.png)

2. Verify that the following packages are listed:
 
![First Four Packages](/docs/images/windows/add_remove_first_4_packs.png)
   
![Last Package](/docs/images/windows/add_remove_last.png)

The exact version numbers may vary depending on what version you're installing, but you must see the following packages listed: **Axelera Cross Compiled Libraries Package**, **Axelera Device Package**, **Axelera Runtime**, **Axelera Services**, and **RISC-V Toolchain and Dependencies for Axelera**.

> [!IMPORTANT]  
> After installing all executable installers, restart your Command Prompt window (without administrator privileges) to continue with the remaining steps.

### Installing the Python Packages
Create a new Python virtual environment for Windows use:
```cmd
python -m venv venv-win
venv-win\Scripts\activate.bat
```
Install the Python packages:
```cmd
pip install windows-packages\axelera_runtime-1.5.0-py3-none-any.whl
pip install windows-packages\axelera_types-1.5.0-py3-none-any.whl
pip install windows-packages\axelera_llm-1.5.0-py3-none-any.whl
```

## Step 5: Running Examples [Optional]
> [!NOTE]  
> If you haven't gone through the previous [WSL step](#step-2-install-the-axelera-voyager-sdk-using-windows-subsystem-for-linux) and need to download a pre-deployed network to do the next steps please run now:
> ```cmd
> cd `C:\Axelera\voyager-sdk`
> axdownloadmodel resnet50-imagenet-onnx
> ```
> The pre-built model will be downloaded under `C:\Axelera\voyager-sdk\build`

Execute inference using either of these methods:

1. **Using Python API**:
   Download an image to `C:\Axelera\voyager-sdk\examples\axruntime\images` and run:
   ```cmd
   cd examples\axruntime
   python -m axruntime_example -v --aipu-cores 1 --labels imagenet-labels.txt ..\..\build\resnet50-imagenet-onnx\resnet50-imagenet-onnx\1\model.json images
   ```
   Where `images` is the directory containing the input images

2. **Using the `axrunmodel` utility to measure performance**:
   ```cmd
   axrunmodel C:\Axelera\voyager-sdk\build\resnet50-imagenet-onnx\resnet50-imagenet-onnx\1\model.json
   ```
   This command will execute the model multiple times and display runtime statistics.

> [!IMPORTANT]  
> Remember to re-activate the Python virtual environment each time you open a new Command Prompt window. From your installation directory (e.g., `C:\Axelera\voyager-sdk`), run:
> ```cmd
> venv-win\Scripts\activate.bat
> ```
> You'll know the virtual environment is active when you see `(venv-win)` at the beginning of your command prompt.

## Step 6: Running LLM Models
It's possible to run a set of precompiled LLM models on Windows.
For the list of available models, please refer to the **LLM Models** section of the [Voyager Model Zoo](/docs/reference/model_zoo.md#large-language-model-llm),
or simply run:
```cmd
axllm --help-network
```

For example, to run the `llama-3-2-1b-1024-4core-static` model in single prompt mode, execute:
```cmd
axllm llama-3-2-1b-1024-4core-static --prompt "Give me a joke"
```

To chat interactively with the model in your terminal, run:
```cmd
axllm llama-3-2-1b-1024-4core-static
```

Many other options are available as described in the [LLM Inference Guide](/docs/tutorials/llm.md#usage-modes).

> [!IMPORTANT]  
> Remember to re-activate the Python virtual environment each time you open a new Command Prompt window. From your installation directory (e.g., `C:\Axelera\voyager-sdk`), run:
> ```cmd
> venv-win\Scripts\activate.bat
> ```
> You'll know the virtual environment is active when you see `(venv-win)` at the beginning of your command prompt.

## Notes on Using Windows PowerShell
After completing all the installation steps above, you can use Windows PowerShell instead of Windows Command Prompt for running examples and working with the SDK. The main difference is in how you activate the Python virtual environment:

```powershell
.\venv-win\Scripts\Activate.ps1
```

Once activated, you can run all the same commands as described in the [Running Examples](#step-5-running-examples-optional) section above.

> [!NOTE]  
> While PowerShell can be used for running examples and working with the SDK after installation, it's important to note that the initial setup steps (particularly Step 4) should still be performed using Windows Command Prompt as specified in the instructions above.
>
> If you encounter issues activating the Python virtual environment in PowerShell, you may need to adjust the PowerShell execution policy. You can check your current policy with:
> ```powershell
> Get-ExecutionPolicy
> ```
> If needed, you can set a more permissive policy using one of these commands (run PowerShell as Administrator):
> ```powershell
> Set-ExecutionPolicy RemoteSigned
> Set-ExecutionPolicy Unrestricted
> ```
> After adjusting the execution policy, try activating the environment again using `.\venv-win\Scripts\Activate.ps1`.

## Next Steps
- **Verify installation**: Run examples to confirm setup
- **Run first inference**: [Quick Start Guide](../quick_start_guide.md)
- **Note**: Firmware updates require temporary Linux access (see [Firmware Update Decision Tree](../firmware_update_decision_tree.md))

## Related Documentation
**Windows Guides:**
- [Windows Voyager SDK Repository](windows_voyager_sdk_repository.md) - Complete FIRST to clone repository
- [Installing Driver](installing_driver.md) - Windows driver installation details
- [Deactivate BitLocker](deactivate_bitlocker.md) - Required before driver installation

**Linux Alternative:**
- [Installation Guide](../install.md) - Linux installation (recommended for best experience)

**Next Steps:**
- [Quick Start Guide](../quick_start_guide.md) - Run first inference after Windows setup
- [Enable Updates](../enable_updates.md) - Enable firmware updates (requires temporary Linux access)

**References:**
- [AxDevice API](../../reference/axdevice.md) - Verify hardware detection after install

## Further support
- For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).
- For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).
