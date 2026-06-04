# Multiple Device Support

## Overview

This guide explains how to use multiple Metis devices with the low-level `axelera.runtime` Python API. A single `Context` object can manage multiple Metis devices in your system, allowing you to scale throughput by distributing workload across multiple accelerators. If you haven't completed the [Quick Start](axruntime-python-quickstart.md) yet, start there to understand basic device connection and resource management.

This guide assumes familiarity with the low-level API's connection and model instance management concepts.

**Prerequisites:**
- Completed [Quick Start](axruntime-python-quickstart.md) and [Basic Usage Tutorial](basic-usage-tutorial.md)
- Multiple Metis devices in your system (e.g., 4-AIPU PCIe card, or two M.2 devices)
- SDK installed: `source venv/bin/activate`

---

## Architecture with Multiple Devices

**Example: 2 Metis devices (8 cores total)**

```
System Resources:
├── Device 0 (4 cores) → Connection 0, 1, 2, 3
└── Device 1 (4 cores) → Connection 4, 5, 6, 7

Single Context manages all:
├── 8 Connections (distributed across devices)
├── 8 ModelInstances (one per Connection)
```

---

## Device Enumeration

### Step 1: List available devices

```python
from axelera.runtime import Context

ctx = Context()
devices = ctx.list_devices()

print(f"Found {len(devices)} device(s)")
for i, device in enumerate(devices):
    print(f"  Device {i}: {device.name}")
```

**Example output:**
```
Found 2 device(s)
  Device 0: metis-0:1:0
  Device 1: metis-0:3:0
```

---

## Pattern 1: Round-Robin Distribution (Simplest)

Distribute connections evenly across devices:

```python
# Key changes from single-device examples:
devices = ctx.list_devices()

# Round-robin assignment across devices
connections = []
for i in range(8):  # 8 cores total
    device_idx = i % len(devices)  # Alternate between devices
    conn = ctx.device_connect(device=devices[device_idx].name, num_sub_devices=1)
    connections.append(conn)
```

**When to use:**
- Simple workloads with one model
- Equal resource distribution is acceptable
- You want balanced load across devices

---

## Pattern 2: Explicit Device Assignment (More Control)

Manually assign connections to specific devices:

```python
# Key changes from axruntime_yolo11.py:
devices = ctx.list_devices()  # Enumerate available devices

# Explicitly target specific devices when creating connections
conn_device0 = ctx.device_connect(device=devices[0].name, num_sub_devices=1)
conn_device1 = ctx.device_connect(device=devices[1].name, num_sub_devices=1)

# Or use device index
conn = ctx.device_connect(device=0, num_sub_devices=1)  # First device
```

**When to use:**
- You need precise control over device allocation
- Testing device-specific behavior
- Debugging hardware issues

---

## Pattern 3: Multi-Device Cascaded Pipeline

Split stages across devices for cascaded inference:

```python
# Key changes from axruntime_cascaded_pipeline.py:
devices = ctx.list_devices()

# Stage 1 on Device 0
stage1_connections = [
    ctx.device_connect(device=devices[0].name, num_sub_devices=1)
    for _ in range(4)
]

# Stage 2 on Device 1
stage2_connections = [
    ctx.device_connect(device=devices[1].name, num_sub_devices=1)
    for _ in range(4)
]
```

**Benefits:**
- Physical isolation between pipeline stages
- No resource contention between models
- Better utilization when stages have different throughput

---

## Best Practices

1. **Single Context for all devices** - Don't create multiple Context objects
2. **Validate device count** - Check `len(ctx.list_devices())` before assuming device availability
3. **Graceful degradation** - Fall back to fewer devices if not all are available
4. **Device affinity** - Consider assigning entire models to single devices for cache locality
5. **Monitor per-device utilization** - Profile each device separately to identify bottlenecks
6. **Balance workload** - Distribute connections evenly unless profiling shows otherwise

---

## Common Patterns

### Pattern: Balanced distribution (recommended)
```python
# 8 cores, 2 devices → 4 cores per device
Device 0: [Conn 0, 1, 2, 3]
Device 1: [Conn 4, 5, 6, 7]
```

### Pattern: Cascade with device isolation
```python
# 8 cores, 2 devices → Stage per device
Device 0: [Stage 1 - all 4 cores]
Device 1: [Stage 2 - all 4 cores]
```

### Pattern: Model-per-device
```python
# 8 cores, 2 devices, 2 models
Device 0: [Model A - 4 instances]
Device 1: [Model B - 4 instances]
```

---

## Example: Adapting Single-Device Code

**Before (single device):**
```python
# Create 4 connections (uses default device)
connections = [
    ctx.device_connect(device=None, num_sub_devices=1)
    for _ in range(4)
]
```

**After (2 devices, round-robin):**
```python
# Enumerate devices
devices = ctx.list_devices()

# Create 8 connections distributed across devices
connections = []
for i in range(8):
    device_idx = i % len(devices)
    conn = ctx.device_connect(
        device=devices[device_idx].name,
        num_sub_devices=1
    )
    connections.append(conn)
```

The rest of the code (loading model instances, running inference) remains the same.

---

## Performance Considerations

**Throughput scaling**: Expect near-linear scaling (2x devices ≈ 2x throughput) for compute-bound workloads.

**PCIe bandwidth**: May become a bottleneck with many devices. Profile to verify.

**CPU bottlenecks**: Ensure preprocessing/postprocessing can keep up with multiple devices.

**Memory**: Each device needs its own set of buffers. Ensure sufficient host memory.

---

## Related Documentation

- [Basic Usage Tutorial](basic-usage-tutorial.md) - Understanding connections and model instances
- [Cascaded Pipelines](cascaded-pipelines.md) - Multi-model applications that can benefit from device isolation
- [API Reference](../../reference/apis/axruntime-py.md) - Context.list_devices() and device_connect()

---

**Last Updated**: 2026-02-11
