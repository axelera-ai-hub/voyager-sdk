# Voyager SDK Makefile
# Copyright Axelera AI, 2025

SHELL := /bin/bash

LOGLEVEL=info
Q=
VERBOSITY=
ifeq ($(LOGLEVEL),error)
  Q=@
  VERBOSITY=-qq
else ifeq ($(LOGLEVEL),warning)
  Q=@
  VERBOSITY=-q
else ifeq ($(LOGLEVEL),info)
  Q=@
else ifeq ($(LOGLEVEL),debug)
  VERBOSITY=-v
else ifeq ($(LOGLEVEL),trace)
  VERBOSITY=-vv
else
  $(error Unknown LOGLEVEL: $(LOGLEVEL) (must be trace, debug, info, warning, or error))
endif

# Debug/release
ifndef CFG
  CFG=release
endif

AXELERA_RUNTIME_DIR  != [ -n "$(AXELERA_RUNTIME_DIR)" ] && echo "$(AXELERA_RUNTIME_DIR)" || python -c "from axelera.runtime.configs import runtime_dir; print(runtime_dir)"

.DEFAULT_GOAL := help

.PHONY: help
help:
	@python3 -c 'from axelera.app import yaml_parser; yaml_parser.gen_model_help()'

# Fast editable reinstall of the Python packages after a pull. Incremental --
# does not clear orphan C++ artifacts; use `rebuild` when extensions change.
.PHONY: develop
develop:
	$(Q)$(MAKE) -C axelera_runtime2 develop
	$(Q)$(MAKE) -C axelera_zoo develop

# Clean rebuild of the Python packages. rt2 recompiles C++ from scratch (drops
# stale/orphan .so in the source tree); use when develop looks out of sync.
.PHONY: rebuild
rebuild:
	$(Q)$(MAKE) -C axelera_runtime2 rebuild
	$(Q)$(MAKE) -C axelera_zoo clean develop

.PHONY: operators
operators: _check-activated-runtime
	AXELERA_RUNTIME_DIR=$(AXELERA_RUNTIME_DIR) $(MAKE) -C operators CFG=$(CFG)

.PHONY: trackers
trackers: _check-activated-runtime
	AXELERA_RUNTIME_DIR=$(AXELERA_RUNTIME_DIR) $(MAKE) -C trackers

.PHONY: examples
examples: _check-activated-runtime operators
	AXELERA_RUNTIME_DIR=$(AXELERA_RUNTIME_DIR) $(MAKE) -C examples

.PHONY: _check-activated-runtime
_check-activated-runtime:
ifeq ($(AXELERA_RUNTIME_DIR),)
	$(error AXELERA_RUNTIME_DIR was not found. Please activate the virtual environment e.g 'source venv/bin/activate' or '/opt/axelera/sdk/<version>/axelera-activate.sh')
endif

.PHONY: clean
clean: clean-libs

.PHONY: clobber
clobber: clobber-libs

.PHONY: clean-libs
clean-libs:
	$(Q)$(MAKE) -C operators clean
	$(Q)$(MAKE) -C trackers clean
	$(Q)$(MAKE) -C examples clean

.PHONY: clobber-libs
clobber-libs:
	$(Q)$(MAKE) -C operators clobber
	$(Q)$(MAKE) -C trackers clobber
	$(Q)$(MAKE) -C examples clobber

.PHONY: operators-docker
operators-docker:
	$(Q)if ! $(MAKE) -sq operators; then \
		echo building operators...; \
		($(MAKE) -C operators clear-cmake-cache &> _operators.log && $(MAKE) operators &>> _operators.log) || \
		echo "Failed to build operators, see _operators.log"; \
	fi

# Variant of operators-docker that clears a stale cmake cache (typical after
# switching venvs). Used by containerless.sh; do not call from install.sh.
.PHONY: operators-containerless
operators-containerless:
	$(Q)for build in operators/Release operators/Debug; do \
		[ -f $$build/CMakeCache.txt ] || continue; \
		old_cmake=$$(awk -F= '/^CMAKE_COMMAND:INTERNAL=/{print $$2; exit}' $$build/CMakeCache.txt); \
		if [ -n "$$old_cmake" ] && [ ! -x "$$old_cmake" ]; then \
			echo "Stale cmake $$old_cmake (likely after venv switch); clearing $$build cache"; \
			rm -f $$build/CMakeCache.txt $$build/build.ninja; \
		fi; \
	done
	$(Q)$(MAKE) operators &> _operators.log || \
		(cat _operators.log; echo "Failed to build operators, see _operators.log"; false)
