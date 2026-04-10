cmake_minimum_required(VERSION 3.10)
project(AxTracker)

# Specify the required C++ standard
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} include)

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
target_link_libraries(axtracker ${OpenCV_LIBS})
target_link_options(axtracker PRIVATE -Wl,--exclude-libs,ALL)
