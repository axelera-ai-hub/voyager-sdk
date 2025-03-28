ifndef CFG
  CFG=release
endif

ifeq ($(CFG),debug)
  BUILD_TYPE=Debug
else
  BUILD_TYPE=Release
endif

BUILD_DIR := $(BUILD_TYPE)
SRC_DIR := src
INCLUDE_DIR := include
CMAKE_FLAGS := -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

all: $(BUILD_DIR)
	ninja -C $(BUILD_DIR) install

$(BUILD_DIR):
	cmake -S . -B $(BUILD_DIR) -G Ninja $(CMAKE_FLAGS) -DCMAKE_INSTALL_PREFIX=$(realpath $(dir $(abspath $(firstword $(MAKEFILE_LIST))))) -DCMAKE_INSTALL_LIBDIR=lib

clean:
	rm -rf Debug Release

clobber: clean

.PHONY: all clean clobber
