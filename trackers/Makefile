ifndef CFG
  CFG=release
endif

ifeq ($(CFG),debug)
  BUILD_TYPE=Debug
else
  BUILD_TYPE=Release
endif

BUILD_DIR := $(BUILD_TYPE)
INSTALL_DIR := lib
CMAKE_FLAGS := -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

CMAKE_CACHE := $(BUILD_DIR)/CMakeCache.txt

define run-cmake
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) -G Ninja $(CMAKE_FLAGS) -DCMAKE_INSTALL_PREFIX=$(realpath $(dir $(abspath $(firstword $(MAKEFILE_LIST))))) -DCMAKE_INSTALL_LIBDIR=$(INSTALL_DIR)
endef

all: $(BUILD_DIR) $(INSTALL_DIR) | check_cmake_rerun
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

$(BUILD_DIR) $(INSTALL_DIR): $(CMAKE_CACHE) | check_cmake_rerun
	ninja -C $(BUILD_DIR) install
	@touch $@

clear-cmake-cache:
	rm -rf $(CMAKE_CACHE)

clean:
ifneq ($(wildcard $(BUILD_DIR)),)
	-ninja clean -C $(BUILD_DIR)
endif
	rm -rf Debug Release

clobber: clean
	rm -rf bin $(INSTALL_DIR)

.PHONY: all clean clobber clear-cmake-cache check_cmake_rerun
