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

.PHONY: operators
operators: _check-activated-runtime
	AXELERA_RUNTIME_DIR=$(AXELERA_RUNTIME_DIR) $(MAKE) -C operators

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
