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

# do var assignments for the specified network, if any
$(foreach ASSIGN,$(shell python3 -c 'from axelera.app import yaml_parser; yaml_parser.gen_model_envs("$(NN)")'),$(eval $(subst _SP_, ,$(ASSIGN))))

ifdef NN
ifeq ($(NN_SET),0)
$(error NN=$(NN) does not specify a valid model. Try 'make help' to print available models)
endif
else
ifneq (,$(filter clean info,$(MAKECMDGOALS)))
  $(error "'$(MAKECMDGOALS)' requires a network to be specified with NN=<network>")
else
ifneq (,$(MAKECMDGOALS))
ifeq (,$(filter clobber-libs clean-libs gst-operators operators trackers examples help,$(MAKECMDGOALS)))
  $(error Type '$(MAKE) help' to get help)
endif
endif
endif
endif

# Compilers
CXX   := gcc
FLAGS := -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Werror

# Debug/release
ifndef CFG
  CFG=release
endif

# Target
HOST_ARCH := $(shell uname -m)

# Build directories (NN_BUILD defined by ASSIGNs above)
NN_BIN_DIR := $(NN_BUILD)/$(CFG)/bin/$(HOST_ARCH)
NN_BIN_SRV := $(NN_BIN_DIR)/$(NN)

PIPELINE   := $(NN_BUILD)/$(HOST_ARCH)/pipeline.h
MODELS     := $(addprefix $(NN_BUILD)/$(HOST_ARCH)/,$(NN_MODELS))

INCLUDE    := $(AXELERA_RUNTIME_DIR)/headers
LIBS       := $(AXELERA_RUNTIME_DIR)/lib

.DEFAULT_GOAL := help

.PHONY: help
help:
	@python3 -c 'from axelera.app import yaml_parser; yaml_parser.gen_model_help()'

.PHONY: info
info:  _check-activated-runtime
	@fold "-w80" <<< "YAML input  : $(NN_FILE)"
	@fold "-w80" <<< "Build dir   : $(NN_BUILD)"
	@fold "-w80" <<< "Description : $(NN_DESC)"

.PHONY: operators gst-operators
operators: _check-activated-runtime
	make -C operators
gst-operators: operators

.PHONY: trackers
trackers: _check-activated-runtime
	make -C trackers

.PHONY: examples
examples: _check-activated-runtime operators
	make -C examples

$(NN_BIN_SRV): $(NN_BIN_DIR)/server.o $(MODELS)
	@mkdir -p "$(@D)"
	$(Q)$(CXX) -L$(LIBS) -Wl,-rpath=$(LIBS) $(FLAGS) -I$(INCLUDE) $< $(MODELS) -o $@ -laxelera-runtime

$(NN_BIN_DIR)/server.o: $(NN_BUILD)/$(HOST_ARCH)/server.c $(PIPELINE)
	$(Q)mkdir -p "$(@D)"
	$(Q)cp server.c $(NN_BIN_DIR)
	$(Q)$(CXX) $(FLAGS) -I$(INCLUDE) -c $< -o $@

$(NN_BUILD)/$(HOST_ARCH)/server.c: server.c
	$(Q)mkdir -p "$(@D)"
	$(Q)cp $< $@

.PHONY: _check-activated-runtime
_check-activated-runtime:
ifeq ($(AXELERA_RUNTIME_DIR),)
	$(error Please set AXELERA_RUNTIME_DIR variable e.g 'source venv/bin/activate')
endif
ifeq ($(AXELERA_FRAMEWORK),)
	$(error Please set AXELERA_FRAMEWORK variable e.g 'source venv/bin/activate')
endif

.PHONY: clean
clean:
	$(Q)$(RM) -r $(NN_BUILD)

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
