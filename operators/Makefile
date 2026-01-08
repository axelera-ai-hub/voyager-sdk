
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
PKG_CONFIG_PATH := $(AXELERA_RUNTIME_DIR)/lib/pkgconfig:$(PKG_CONFIG_PATH)

INSTALL_DIR := $(PREFIX_DIR)/lib

ARCHX := $(shell uname -m)
ifeq ($(ARCHX),arm64)
ARCH := $(shell uname -p)
ifeq ($(ARCH),arm64)
  ONNX_RUNTIME_ARCH := aarch64
else
ifeq ($(ARCH),aarch64)
  ONNX_RUNTIME_ARCH := aarch64
else
ifeq ($(ARCH),arm)
  ONNX_RUNTIME_ARCH := aarch64
else
  $(error Unsupported architecture: $(ARCH) "$(MACHINE_ARCH)")
endif
endif
endif
else
ifeq ($(ARCHX),x86_64)
ARCH2 := $(shell uname -m)
ifeq ($(ARCH2),x86_64)
  ONNX_RUNTIME_ARCH := x64
else
  $(error Unsupported architecture: $(ARCH2) "$(MACHINE_ARCH)")
endif
endif
endif
OS := $(shell uname -s)
ifeq ($(OS),Linux)
  ONNX_RUNTIME_OS := linux
else
ifeq ($(OS),Darwin)
  ONNX_RUNTIME_OS := osx
  ONNX_RUNTIME_ARCH := arm64
else
  $(error Unsupported OS: $(OS))
endif
endif
ONNX_RUNTIME_VERSION := 1.17.1
ONNX_RUNTIME_BASEDIR := onnxruntime
ONNX_RUNTIME_NAME := onnxruntime-$(ONNX_RUNTIME_OS)-$(ONNX_RUNTIME_ARCH)-$(ONNX_RUNTIME_VERSION)
ONNX_RUNTIME_URL := https://github.com/microsoft/onnxruntime/releases/download/v$(ONNX_RUNTIME_VERSION)/$(ONNX_RUNTIME_NAME).tgz
ONNX_RUNTIME_DIR := $(ONNX_RUNTIME_BASEDIR)/$(ONNX_RUNTIME_NAME)

CMAKE_CACHE := $(BUILD_DIR)/CMakeCache.txt

define run-cmake
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
	PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) cmake .. $(1) -GNinja -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_INSTALL_PREFIX=$(PREFIX_DIR) -DCMAKE_INSTALL_LIBDIR=lib  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON || (cd .. && $(call clobber-it) && exit 1)
endef

gst_ops_install:  $(BUILD_DIR) $(ONNX_RUNTIME_DIR) $(INSTALL_DIR) | check_cmake_rerun
# If build directory doesn't exist, its prerequisites will create it and run ninja
# If it exists, only run ninja if there's work to do
ifneq ($(wildcard $(BUILD_DIR)),)
ifeq (,$(findstring no work to do,$(shell ninja -n -C $(BUILD_DIR))))
	ninja -C $(BUILD_DIR) install
endif
endif

check_cmake_rerun:
ifneq ($(wildcard $(BUILD_DIR)),)
ifneq (,$(findstring build.ninja older,$(shell ninja -n -d explain -C $(BUILD_DIR) 2>&1)))
	$(call run-cmake,--fresh)
endif
endif

$(CMAKE_CACHE):
	$(call run-cmake)
	
$(BUILD_DIR) $(INSTALL_DIR): $(ONNX_RUNTIME_DIR) $(CMAKE_CACHE) | check_cmake_rerun
	ninja -C $(BUILD_DIR) install
	@touch $@

$(ONNX_RUNTIME_DIR):
	mkdir -p $(ONNX_RUNTIME_BASEDIR)
	(cd $(ONNX_RUNTIME_BASEDIR) && wget $(ONNX_RUNTIME_URL) && tar -xzf $(ONNX_RUNTIME_NAME).tgz)  || (cd .. && $(call clobber-it) && exit 1)
	ln -snf $(ONNX_RUNTIME_NAME) onnxruntime/onnxruntime
	mkdir -p lib
	for lib in onnxruntime/onnxruntime/lib/*; do ln -snf "../$${lib}" lib; done
	
clear-cmake-cache:
	rm -rf $(CMAKE_CACHE)
	
clean:
	-ninja clean -C $(BUILD_DIR)
	
define clobber-it
	rm -rf $(BUILD_DIR) $(INSTALL_DIR)  $(ONNX_RUNTIME_DIR) $(ONNX_RUNTIME_BASEDIR)/$(ONNX_RUNTIME_NAME).tgz $(ONNX_RUNTIME_BASEDIR)/onnxruntime
endef
	
clobber:
	$(call clobber-it)
	
.phony: clean clobber gst_ops_install check_cmake_rerun
