cmake_minimum_required(VERSION 3.12)
project(
  PillowResize
  VERSION 1.0.0
  LANGUAGES CXX)

#
# BUILD SETTINGS
#

# Set compilation flags
if(NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        "Release"
        CACHE STRING "" FORCE)
  endif()

endif()

SET (CMAKE_CXX_FLAGS "-ftree-vectorize -march=native -O2")

# (un)set: cmake -DCVV_DEBUG_MODE=OFF ..
option(CVV_DEBUG_MODE "cvvisual-debug-mode" OFF)
if(CVV_DEBUG_MODE MATCHES ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCVVISUAL_DEBUGMODE")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

#
# BUILD DEPENDENCIES
#
# Build and setup the correct cmake variables for third-party libraries
#

# Find required libraries
find_package(OpenCV REQUIRED)

# updated in the following steps
set(RESIZE_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include/PillowResize")

#
# Build project
#
add_subdirectory(src/PillowResize)

# the variable for the PillowResize libraries
set(PILLOWRESIZE_LIBS PillowResize)

# Helpers functions for creating config files that can be included by other
# projects to find and use a package. Adds the configure_package_config_file()
# and write_basic_package_version_file() commands.
include(CMakePackageConfigHelpers)

# Create a ConfigVersion.cmake file
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

