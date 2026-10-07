---
title: "axelera.runtime.device_config"
---
# `axelera.runtime.device_config`


Device settings a model asks a board for before it runs.

Keyed by board MODEL (`110m`), not form factor (`m2`): a 110m and a 113m are
both `m2`, so a form factor cannot say which of them a card's value was measured
on. `axdevice` resolves the model and owns that mapping.

Applied per AI core, to the cores running that model, as the YAML/GStreamer path
does with its static core ranges. Unlike that path, which binds a core to a task
for the whole run and so writes each setting once, this scheduler shares cores,
so a setting is rewritten whenever a core changes to a model that wants a
different value.

Three sources, first that answers wins: `AXELERA_CONFIGURE_BOARD` (see
`env_override`), then `scheduler.create(device_config=...)`, then the model's own
`op.load(device_config=...)`. A model's config travels with the artifact; the
scheduler argument belongs to the deployment; the variable belongs to whoever
runs the binary.

## Summary

| Name | Description |
|------|-------------|
| [board_models](#board_models) | Every board model a `DeviceConfig` may name, from `axdevice`'s own table. |
| [BoardConfig](#boardconfig) | Settings a model wants on one board model; one field per `SETTINGS` entry. |
| [DeviceConfig](#deviceconfig) | Settings a model wants, per board model. |
| [Setting](#setting) | One setting's meaning, written once and rendered into generated recipes. |
| [board_model](#board_model) | Return the board model `axdevice` reports, or a form factor it cannot place. |
| [env_override](#env_override) | Read `AXELERA_CONFIGURE_BOARD`, returning `(configure, clock, mvm)`. |

---

### board_models

```python
board_models() -> frozenset[str]
```

Every board model a `DeviceConfig` may name, from `axdevice`'s own table.

Form factors are excluded: `m2` names both the 110m and the 113m. Empty
without `axelera-runtime`, which leaves names unchecked on a machine that
could not act on them anyway.

---

### BoardConfig

Settings a model wants on one board model; one field per `SETTINGS` entry.

**Args:**

- **mvm_limitation**: see `SETTINGS`. Hardware quantises to ~1.5% steps.

**Constructor:**

```python
__init__(mvm_limitation: int | None = None)
```

---

### DeviceConfig

Settings a model wants, per board model. A board not named is left alone.

**Args:**

- **boards**: board model (one of `board_models()`) to its settings. An unknown name raises, so a typo fails where it is written.

**Example:**

```python
limits = DeviceConfig({'110m': BoardConfig(mvm_limitation=57)})
op.load('yolov8l-pose-coco.axm', device_config=limits)
```

**Constructor:**

```python
__init__(boards: dict[str, BoardConfig] = field(default_factory=dict))
```

**Methods:**

#### as_spec

```python
as_spec() -> dict[str, dict[str, int]]
```

Plain data for a manifest. TOML has no null, so an unset value is absent.

#### classmethod from_spec

```python
from_spec(spec: dict[str, dict[str, int]]) -> DeviceConfig
```

Rebuild a DeviceConfig from `as_spec` output.

#### explain

```python
explain() -> list[str]
```

Return the reason recorded for each setting in use, for a generated recipe.

#### for_model

```python
for_model(model: str | None) -> BoardConfig | None
```

Return the entry for a board model, or None if this config names none.

---

### Setting

One setting's meaning, written once and rendered into generated recipes.

**Args:**

- **summary**: what the setting is and what values it takes.
- **boards**: board models a packaged model declares it for. Advisory, not a support claim.
- **reason**: optional, why. Printed above a recipe's DEVICE_CONFIG.

**Constructor:**

```python
__init__(summary: str, boards: tuple[str, ...] = (), reason: str = '')
```

---

### board_model

```python
board_model(device) -> str | None
```

Return the board model `axdevice` reports, or a form factor it cannot place.

---

### env_override

```python
env_override() -> tuple[bool, int | None, int | None]
```

Read `AXELERA_CONFIGURE_BOARD`, returning `(configure, clock, mvm)`.

The grammar is the one the YAML/GStreamer path already documents, so one
variable governs both paths:

- `0`: do not configure the board at all. `configure` comes back False and
  the model's own request is dropped.
- `1` (the default, and what an unset variable means): configure the board
  from the model's request. Both overrides come back `None`.
- `[clock],[mvm]`: the user's own values, either part may be empty. A value
  given here replaces the model's, so `,90` caps MVM at 90% whatever the
  model asked for and `400` sets the clock and leaves MVM to the model.

**Raises:**

- **ValueError**: if the variable is set to something this grammar does not read.
