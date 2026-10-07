cmake_minimum_required(VERSION 3.10)
project(AxTracker)

# Specify the required C++ standard
set(CMAKE_CXX_STANDARD 17)

# Parent handles: find_package(OpenCV/Eigen3), compiler options, install

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
target_link_libraries(axtracker PUBLIC ${OpenCV_LIBS})
target_link_options(axtracker PRIVATE ${AX_TRACKER_LINK_OPTIONS})
