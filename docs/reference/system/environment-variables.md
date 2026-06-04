---
title: "Environment Variables"
---
# Environment Variables

When you run `source venv/bin/activate`, the SDK sets several environment variables that tell its tools where to find libraries, firmware, and device binaries. Understanding these helps when troubleshooting or deploying to non-standard locations.

## Variables set by activation

| Variable | Default value | Purpose |
|----------|--------------|---------|
| `AXELERA_FRAMEWORK` | `.` (SDK root) | Location of the Voyager SDK repository |
| `AXELERA_RUNTIME_DIR` | `/opt/axelera/runtime-<version>/omega` | Host runtime libraries for Metis devices |
| `AXELERA_DEVICE_DIR` | `/opt/axelera/device-<version>` | Metis firmware and low-level binaries |
| `LD_LIBRARY_PATH` | `$AXELERA_RUNTIME_DIR/lib:$AXELERA_FRAMEWORK/operators/lib` | Dynamic library search path |
| `PYTHONPATH` | `$AXELERA_RUNTIME_DIR/tvm/tvm-src` | Python module search path for SDK packages |

## What each variable does

### AXELERA_FRAMEWORK

Points to the SDK repository root. All tools use this to find models, pipelines, and configuration files.

```bash
echo $AXELERA_FRAMEWORK
# /home/user/voyager-sdk
```

### AXELERA_RUNTIME_DIR

Location of the host-side runtime libraries. These are installed by `install.sh` to `/opt/axelera/` in version-specific directories.

```bash
echo $AXELERA_RUNTIME_DIR
# /opt/axelera/runtime-1.5.0/omega
```

### AXELERA_DEVICE_DIR

Location of the device firmware and low-level binaries that are loaded onto the Metis [AIPU](../../glossary.md#aipu) at runtime.

```bash
echo $AXELERA_DEVICE_DIR
# /opt/axelera/device-1.5.0
```

### LD_LIBRARY_PATH

Standard Linux variable that tells the dynamic linker where to find shared libraries. The SDK adds its own library paths so that compiled models can access the Axelera runtime.

### PYTHONPATH

Standard Python variable that adds directories to the module search path. The SDK adds paths so that `import axelera` and related imports work.

## Useful for troubleshooting

If a tool reports missing libraries or modules, check these variables:

```bash
# Verify environment is active
echo $AXELERA_FRAMEWORK    # Should show your SDK path
echo $AXELERA_RUNTIME_DIR  # Should show /opt/axelera/runtime-...

# If empty, you forgot to activate
source venv/bin/activate
```

## Optional variables

| Variable | Purpose |
|----------|---------|
| `AXELERA_CONFIGURE_BOARD=0` | Prevent pipelines from auto-configuring device clock/MVM settings |

## After deactivation

When you run `deactivate`, all variables are restored to their original values (or unset if they didn't exist before activation). This prevents SDK paths from leaking into unrelated work.

## See also

- [Virtual Environments](virtual-environments.md) — why activation is required
- [axdevice](../tools/axdevice.md) — uses these variables to find device firmware
- [install.sh](../tools/install-sh.md) — creates the paths these variables point to
