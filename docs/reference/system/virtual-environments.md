---
title: "Virtual Environments"
---
# Virtual Environments — Why and How

A Python virtual environment is an isolated space where Python packages are installed without affecting your system Python. The Voyager SDK uses one to keep its dependencies separate from everything else on your machine.

## Why it matters

The SDK depends on specific versions of hundreds of Python packages. Without a virtual environment:

- Installing the SDK could break other Python tools on your system
- Other tools could break the SDK by upgrading a shared dependency
- Multiple SDK versions would conflict with each other

The virtual environment solves all of this. Everything the SDK needs lives in `venv/`.

## The golden rule

> [!CAUTION]
> **You must activate the environment every time**
> ```bash
> source venv/bin/activate
> ```
>
> Run this **every time** you open a new terminal before using any SDK tool. There are no exceptions. If you see errors like `command not found` or `ModuleNotFoundError`, you forgot to activate.


## How to tell if it's active

When the environment is active, your terminal prompt changes:

```
# Inactive — no prefix
user@host:~/voyager-sdk$

# Active — (venv) prefix
(venv) user@host:~/voyager-sdk$
```

If you don't see `(venv)`, the environment is not active.

## Common operations

| Action | Command |
|--------|---------|
| Activate | `source venv/bin/activate` |
| Deactivate | `deactivate` |
| Check if active | Look for `(venv)` in your prompt |
| Check Python location | `which python` — should point to `venv/bin/python` |

## Where it lives

The SDK installer creates the virtual environment at:

```
~/.cache/axelera/venvs/<sdk-version>/
```

A symlink at `./venv` in the SDK directory points to it. This means:

- `source venv/bin/activate` works from the SDK root
- The actual files are in your home directory cache
- Multiple SDK versions each have their own environment

## Common mistakes

### 1. Forgot to activate

**Symptom:** `command not found: inference.py` or `ModuleNotFoundError`

**Fix:** `source venv/bin/activate`

### 2. Wrong environment active

**Symptom:** Unexpected behavior, wrong model versions, missing features

**Fix:** `deactivate` then `source venv/bin/activate` from the correct SDK directory

### 3. Activated from the wrong directory

**Symptom:** Environment activates but SDK tools don't work

**Fix:** Make sure you're in the SDK root directory (where `venv/` is a symlink) before activating

### 4. System Python used instead

**Symptom:** `which python` shows `/usr/bin/python` instead of a `venv/` path

**Fix:** `deactivate` (if another env is active), then `source venv/bin/activate`

## Switching SDK versions

When switching between SDK versions:

```bash
# Step 1: Deactivate current environment
deactivate

# Step 2: Move to the other SDK directory
cd ~/sdk-v1.4

# Step 3: Activate that version's environment
source venv/bin/activate
```

Never activate one version's environment from another version's directory.

## See also

- [SDK Install](../../user-guides/sdk-install.md) — where activation happens in the install process
- [Environment Variables](environment-variables.md) — what activation sets behind the scenes
- [install.sh](../tools/install-sh.md) — the installer that creates the environment
