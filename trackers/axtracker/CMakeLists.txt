cmake_minimum_required(VERSION 3.10)
project(AxTracker)

# Specify the required C++ standard
set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED IMPORTED_TARGET opencv4)
if(NOT OpenCV_INCLUDE_DIRS)
  pkg_get_variable(OpenCV_INCLUDEDIR opencv4 includedir)
  if(NOT OpenCV_INCLUDEDIR)
    message(FATAL_ERROR "opencv4 include path is empty")
  endif()
  if(EXISTS "${OpenCV_INCLUDEDIR}/opencv2/core.hpp")
    set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDEDIR})
  elseif(EXISTS "${OpenCV_INCLUDEDIR}/opencv4/opencv2/core.hpp")
    set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDEDIR}/opencv4)
  else()
    message(FATAL_ERROR "Cannot locate OpenCV headers under ${OpenCV_INCLUDEDIR}")
  endif()
endif()

# Include directories
include_directories(include ${OpenCV_INCLUDE_DIRS})

# Source files
file(GLOB AXTRACKER_SOURCES "src/*.cpp")

# Add axtracker library
add_library(axtracker SHARED ${AXTRACKER_SOURCES})

# Set RPATH to $ORIGIN so library can find co-located dependencies
set_target_properties(axtracker PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_RPATH "$ORIGIN"
)

# Link libraries
target_link_libraries(axtracker PkgConfig::OpenCV)
target_link_options(axtracker PRIVATE -Wl,--exclude-libs,ALL)
