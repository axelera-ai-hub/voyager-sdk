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

# Builds operators, clearing a stale cmake cache (typical after switching
# venvs). Used by containerless.sh.
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

PYTHON_VERSIONS ?= 3.10 3.11 3.12 3.13 3.14
REQUIREMENT_SPECS := \
	requirements.axelera.in:requirements.axelera.txt:requirements \
	requirements.runtime.in:requirements.runtime.txt:requirements \
	requirements.unittests.in:requirements.unittests.txt:requirements \
	requirements.CI.in:requirements.CI.txt:requirements \
	axelera_llm/requirements.build.ci.in:requirements.build.ci.txt:axelera_llm/requirements \
	axelera_llm/requirements.tests.in:requirements.tests.txt:axelera_llm/requirements
REQUIREMENT_INPUTS := requirements.application.txt requirements.operators.txt $(foreach spec,$(REQUIREMENT_SPECS),$(word 1,$(subst :, ,$(spec))))
REQUIREMENT_OUTPUTS := $(foreach py,$(PYTHON_VERSIONS),$(foreach spec,$(REQUIREMENT_SPECS),$(word 3,$(subst :, ,$(spec)))/py$(py)/$(word 2,$(subst :, ,$(spec)))))
REQUIREMENT_MANIFEST := requirements/requirements.sha256

.PHONY: requirements check-requirements check-requirement-ranges clean-requirements
requirements:
	@set -e; \
	req_tmp=$$(mktemp -d "$${TMPDIR:-/tmp}/axelera-requirements.XXXXXX"); \
	trap 'rm -rf "$$req_tmp"' EXIT; \
	uv python install --force $(PYTHON_VERSIONS); \
	for py in $(PYTHON_VERSIONS); do \
		venv="$$req_tmp/py$$py"; \
		uv venv --clear --seed --python "$$py" "$$venv"; \
		"$$venv/bin/python" -m pip install --upgrade pip pip-tools; \
		for spec in $(REQUIREMENT_SPECS); do \
			input=$${spec%%:*}; rest=$${spec#*:}; output=$${rest%%:*}; package=$${rest#*:}; \
			mkdir -p "$$package/py$$py"; \
			PIP_CONFIG_FILE=pip.conf "$$venv/bin/pip-compile" --resolver=backtracking --output-file="$$package/py$$py/$$output" "$$input"; \
		done; \
	done; \
	sha256sum $(REQUIREMENT_INPUTS) $(REQUIREMENT_OUTPUTS) > $(REQUIREMENT_MANIFEST)

check-requirements:
	@if [[ ! -f $(REQUIREMENT_MANIFEST) ]] || ! sha256sum --check --status $(REQUIREMENT_MANIFEST); then \
		echo "Requirements sources or generated lockfiles differ from $(REQUIREMENT_MANIFEST)."; \
		echo "Ensure any required edits have been done to the source .in file(s) not the .txt file(s) and run 'make requirements'."; \
		exit 1; \
	fi

check-requirement-ranges:
	@set -e; \
	python3 ci-scripts/check_requirement_ranges.py; \
	command -v uv >/dev/null || { \
		echo "uv is required to resolve requirement ranges; see https://docs.astral.sh/uv/"; \
		exit 1; \
	}; \
	for py in $(PYTHON_VERSIONS); do \
		echo "Resolving requirement ranges for Python $$py..."; \
		log=$$(mktemp); \
		if ! PIP_CONFIG_FILE=pip.conf uv run --no-project --python "$$py" \
			--with pip-tools pip-compile --quiet --dry-run --resolver=backtracking --output-file=/dev/null \
			--extra-index-url https://download.pytorch.org/whl/cpu \
					requirements.operators.txt >"$$log" 2>&1; then \
			echo "Requirement range resolution failed for Python $$py:"; \
			grep -E 'ERROR:|DistributionNotFound' "$$log" | head -2 || tail -5 "$$log"; \
			echo "Relevant application requirements:"; \
			while IFS= read -r requirement; do \
				case "$$requirement" in ''|\#*|--*) continue;; esac; \
				name=$$(printf '%s\n' "$$requirement" | sed -E 's/^([A-Za-z0-9_.-]+).*/\1/'); \
				grep -iE "^$${name}([[:space:]]|[<>=!~;]|$$)" requirements.application.txt || true; \
			done < requirements.operators.txt | sort -fu; \
			echo "Relevant operator requirements:"; \
			grep -Ev '^[[:space:]]*(#|$$|--)' requirements.operators.txt || true; \
			rm -f "$$log"; \
			exit 1; \
		fi; \
		rm -f "$$log"; \
	done; \
	echo "Requirement range checks passed."

clean-requirements:
	rm -rf .venvs/req-* requirements/py* axelera_llm/requirements/py* $(REQUIREMENT_MANIFEST)
