# Voyager SDK Makefile for example C++ applications
# Copyright Axelera AI, 2025

ifndef CFG
  CFG=release
endif

ifeq ($(CFG),debug)
  BUILD_TYPE=Debug
else
  BUILD_TYPE=Release
endif

BUILD_DIR := $(BUILD_TYPE)

PREFIX_DIR := $(realpath $(dir $(abspath $(firstword $(MAKEFILE_LIST)))))
PKG_CONFIG_PATH := $(AXELERA_RUNTIME_DIR)/lib/pkgconfig:$(AXELERA_FRAMEWORK)/operators/lib/pkgconfig:$(PKG_CONFIG_PATH)


gst_ops_install:  $(BUILD_DIR)
	ninja -C $(BUILD_DIR) install

$(BUILD_DIR):
	# mkdir -p $(BUILD_DIR)
	PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) cmake -B$(BUILD_DIR) -GNinja . -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_INSTALL_PREFIX=$(PREFIX_DIR)  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON || ($(call clobber-it) && exit 1)

clean:
	-ninja clean -C $(BUILD_DIR)

define clobber-it
	rm -rf $(BUILD_DIR)
endef

clobber:
	$(call clobber-it)

.phony: clean clobber
