
ifndef CFG
  CFG=release
endif

# CUDA support is disabled by default
# To enable CUDA, explicitly set CUDA_AVAILABLE=1
# Requirements: nvidia GPU (nvidia-smi), cuda toolkit (nvcc), and libcudnn8-dev
ifndef CUDA_AVAILABLE
  CUDA_AVAILABLE=0
else
  $(info CUDA_AVAILABLE explicitly set to $(CUDA_AVAILABLE))
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

ARCH := $(shell uname -m)
ifeq ($(ARCH),arm64)
  ONNX_RUNTIME_ARCH := aarch64
else
ifeq ($(ARCH),aarch64)
  ONNX_RUNTIME_ARCH := aarch64
else
ifeq ($(ARCH),arm)
  ONNX_RUNTIME_ARCH := aarch64
else
ifeq ($(ARCH),x86_64)
  ONNX_RUNTIME_ARCH := x64
else
  $(error Unsupported architecture: $(ARCH) "$(MACHINE_ARCH)")
endif
endif
endif
endif

OS := $(shell uname -s)
ifeq ($(OS),Linux)
  ONNX_RUNTIME_OS := linux
  SHA256SUM := sha256sum
else
ifeq ($(OS),Darwin)
  ONNX_RUNTIME_OS := osx
  ONNX_RUNTIME_ARCH := arm64
  SHA256SUM := shasum -a 256
else
  $(error Unsupported OS: $(OS))
endif
endif
ONNX_RUNTIME_VERSION := 1.17.1
ONNX_RUNTIME_BASEDIR := onnxruntime
# ONNX Runtime comes from Microsoft's NuGet packages, which carry the same binaries as
# the GitHub release tarballs. The tree under $(ONNX_RUNTIME_DIR) keeps the tarball layout.
ifeq ($(CUDA_AVAILABLE),1)
  ONNX_RUNTIME_DIRNAME := onnxruntime-$(ONNX_RUNTIME_OS)-$(ONNX_RUNTIME_ARCH)-cuda-$(ONNX_RUNTIME_VERSION)
  ONNX_RUNTIME_PACKAGE := microsoft.ml.onnxruntime.gpu.linux
  # nuget.org has the CUDA 11 build of this version; Microsoft's CUDA 12 feed has the CUDA 12 one.
  ONNX_RUNTIME_FEED := https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/nuget/v3/flat2
  ONNX_RUNTIME_SHA256 := 3b8710e919c64443be4ec72662c71c056d49ebed91662193baef1d215c50720b
  ONNX_RUNTIME_INCLUDE := buildTransitive/native/include
else
  ONNX_RUNTIME_DIRNAME := onnxruntime-$(ONNX_RUNTIME_OS)-$(ONNX_RUNTIME_ARCH)-$(ONNX_RUNTIME_VERSION)
  ONNX_RUNTIME_PACKAGE := microsoft.ml.onnxruntime
  ONNX_RUNTIME_FEED := https://api.nuget.org/v3-flatcontainer
  ONNX_RUNTIME_SHA256 := 60493669633f99542a0a3e4dd17bfa9542cbfcd9852ef672f8c36ce6a9d2d935
  ONNX_RUNTIME_INCLUDE := build/native/include
endif
ONNX_RUNTIME_NAME := $(ONNX_RUNTIME_PACKAGE).$(ONNX_RUNTIME_VERSION)
ONNX_RUNTIME_URL := $(ONNX_RUNTIME_FEED)/$(ONNX_RUNTIME_PACKAGE)/$(ONNX_RUNTIME_VERSION)/$(ONNX_RUNTIME_NAME).nupkg
# NuGet runtime identifier of the library to take from the package.
ONNX_RUNTIME_RID := $(ONNX_RUNTIME_OS)-$(subst aarch64,arm64,$(ONNX_RUNTIME_ARCH))
ifeq ($(OS),Darwin)
  ONNX_RUNTIME_LIB := libonnxruntime.dylib
  ONNX_RUNTIME_LIB_VERSIONED := libonnxruntime.$(ONNX_RUNTIME_VERSION).dylib
else
  ONNX_RUNTIME_LIB := libonnxruntime.so
  ONNX_RUNTIME_LIB_VERSIONED := libonnxruntime.so.$(ONNX_RUNTIME_VERSION)
endif
ONNX_RUNTIME_DIR := $(ONNX_RUNTIME_BASEDIR)/$(ONNX_RUNTIME_DIRNAME)

CMAKE_CACHE := $(BUILD_DIR)/CMakeCache.txt

define run-cmake
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
	PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) cmake .. $(1) -GNinja -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_INSTALL_PREFIX=$(PREFIX_DIR) -DCMAKE_INSTALL_LIBDIR=lib -DCUDA_AVAILABLE=$(CUDA_AVAILABLE) || \
		(cd .. && \
		echo "\n*** Some platforms may have different/extra package requirements not covered by SDK installation." && \
		echo "*** If cmake has reported missing packages you may need to identify and install these separately.\n" && \
		$(call clobber-it) && exit 1)
endef

gst_ops_install:  $(BUILD_DIR) $(ONNX_RUNTIME_DIR) $(INSTALL_DIR) | check_cmake_rerun
# If build directory doesn't exist, its prerequisites will create it and run ninja
# If it exists, only run ninja if there's work to do
ifneq ($(wildcard $(BUILD_DIR)),)
ifeq (,$(findstring no work to do,$(shell ninja -n -C $(BUILD_DIR))))
	ninja -C $(BUILD_DIR) install -j ${shell nproc --all --ignore=2}
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
	ninja -C $(BUILD_DIR) install -j ${shell nproc --all --ignore=2}
	@touch $@

# The tree is assembled in $(ONNX_RUNTIME_STAGE) and renamed last, so a failed step
# never leaves a $(ONNX_RUNTIME_DIR) that make would take as complete.
# The package headers have CRLF line endings; they are stored with LF like the tarball's.
ONNX_RUNTIME_STAGE := $(ONNX_RUNTIME_DIRNAME).part
$(ONNX_RUNTIME_DIR):
	mkdir -p $(ONNX_RUNTIME_BASEDIR)
	(cd $(ONNX_RUNTIME_BASEDIR) && rm -rf $(ONNX_RUNTIME_STAGE) && \
		wget -O $(ONNX_RUNTIME_NAME).nupkg $(ONNX_RUNTIME_URL) && \
		echo "$(ONNX_RUNTIME_SHA256)  $(ONNX_RUNTIME_NAME).nupkg" | $(SHA256SUM) -c - && \
		mkdir -p $(ONNX_RUNTIME_STAGE)/include && \
		unzip -q -j $(ONNX_RUNTIME_NAME).nupkg '$(ONNX_RUNTIME_INCLUDE)/*.h' -d $(ONNX_RUNTIME_STAGE)/crlf && \
		for h in $(ONNX_RUNTIME_STAGE)/crlf/*.h; do \
			tr -d '\r' < "$$h" > "$(ONNX_RUNTIME_STAGE)/include/$${h##*/}" || exit 1; done && \
		rm -rf $(ONNX_RUNTIME_STAGE)/crlf && \
		unzip -q -j $(ONNX_RUNTIME_NAME).nupkg 'runtimes/$(ONNX_RUNTIME_RID)/native/*' -d $(ONNX_RUNTIME_STAGE)/lib && \
		mv $(ONNX_RUNTIME_STAGE)/lib/$(ONNX_RUNTIME_LIB) $(ONNX_RUNTIME_STAGE)/lib/$(ONNX_RUNTIME_LIB_VERSIONED) && \
		ln -s $(ONNX_RUNTIME_LIB_VERSIONED) $(ONNX_RUNTIME_STAGE)/lib/$(ONNX_RUNTIME_LIB) && \
		unzip -q -j $(ONNX_RUNTIME_NAME).nupkg LICENSE ThirdPartyNotices.txt -d $(ONNX_RUNTIME_STAGE) && \
		mv $(ONNX_RUNTIME_STAGE) $(ONNX_RUNTIME_DIRNAME)) || (cd .. && $(call clobber-it) && exit 1)
	ln -snf $(ONNX_RUNTIME_DIRNAME) onnxruntime/onnxruntime
	mkdir -p lib
	for lib in onnxruntime/onnxruntime/lib/*; do ln -snf "../$${lib}" lib; done

clear-cmake-cache:
	rm -rf $(CMAKE_CACHE)

clean:
	-ninja clean -C $(BUILD_DIR)

define clobber-it
	rm -rf $(BUILD_DIR) $(INSTALL_DIR)  $(ONNX_RUNTIME_DIR) $(ONNX_RUNTIME_BASEDIR)/$(ONNX_RUNTIME_NAME).nupkg $(ONNX_RUNTIME_BASEDIR)/$(ONNX_RUNTIME_STAGE) $(ONNX_RUNTIME_BASEDIR)/onnxruntime
endef

clobber:
	$(call clobber-it)

.phony: clean clobber gst_ops_install check_cmake_rerun
