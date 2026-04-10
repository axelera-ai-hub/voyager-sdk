![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Voyager SDK Repository on Windows

## Contents
- [Voyager SDK Repository on Windows](#voyager-sdk-repository-on-windows)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Level](#level)
  - [Overview](#overview)
  - [Installation Steps](#installation-steps)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Further support](#further-support)

## Prerequisites
- Windows 11
- Git for Windows or equivalent Git client
- Internet connection

## Level
**Beginner** - Repository cloning and documentation access

## Overview
The Voyager SDK repository and the associated tools are currently designed for Linux systems.
However, the repository itself contains useful documentation and examples that can be used on Windows.
This guide provides a basic overview of how to access and browse the Voyager SDK repository on a Windows system without WSL. For the full Voyager SDK experience that includes local Neural Network deployment, WSL is the recommended approach
(see the [Windows Getting Started](/docs/tutorials/windows/windows_getting_started.md) guide).




## Installation Steps
> [!NOTE]  
> To clone the Voyager SDK repository, you'll need a version of Git installed on Windows. 
> Any Git implementation will work: MSYS2 Git, Git for Windows (available at [git-scm.com](https://git-scm.com/download/win)), GitHub Desktop, Git through Visual Studio or any other Windows git option. 
> The choice is entirely up to your preference.

1. **Clone the Repository**: Open a command prompt or PowerShell window and run the following command to clone the Voyager SDK repository:

```cmd
git clone https://github.com/axelera-ai-hub/voyager-sdk.git
```

2. **Navigate to the Repository**: Change into the cloned directory:

```cmd
cd voyager-sdk
```

The head of the repository is always set to the latest published SDK release. You can use
standard git commands to list the available releases and to checkout different versions of the
SDK. Run the following command to  view the current release branch and all available releases:

```bash
git branch
```

To checkout a specific SDK release, run a command such as:

```
git checkout release/v1.6.0
git rebase
```

To rebase to the latest publicly released SDK version, run the following command:

```bash
git checkout main
git rebase
```

3. **Explore the Repository**: You can now explore the contents of the repository. The Windows relevant files and directories are as follows:

| File or directory | Description |
| :---------------- | :---------- |
| [`examples/`](/examples) | Example applications utilizing different models and pipelines |
| [`docs`](/docs/) | Tutorial and reference documentation |
| [`licenses/`](/licenses) | Licenses for all SDK components and dependencies |

## Next Steps
- **After cloning**: Proceed to [Windows Getting Started](windows_getting_started.md)
- **For full SDK**: Complete WSL2 setup as described in Windows Getting Started
- **Browse documentation**: Navigate repository for examples and guides

- [Voyager SDK Repository on Windows](#voyager-sdk-repository-on-windows)
  - [Overview](#overview)
  - [Installation Steps](#installation-steps)

## Related Documentation
**Windows Guides:**
- [Windows Getting Started](windows_getting_started.md) - Complete this AFTER cloning repository
- [Installing Driver](installing_driver.md) - Driver installation after repository setup
- [Deactivate BitLocker](deactivate_bitlocker.md) - Required before driver installation

**Linux Alternative:**
- [Installation Guide](../install.md) - Linux installation (recommended for full SDK experience)

## Further support

For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).

For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).
