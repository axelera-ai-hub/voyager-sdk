cmake_minimum_required(VERSION 3.10)
project(Trackers)

set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include directories for the entire project
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS}
)

# Add subdirectories
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/algorithms/bytetrack")
  message(STATUS "Building bytetrack")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/algorithms/bytetrack")
  add_definitions(-DHAVE_BYTETRACK)
endif()

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/algorithms/oc_sort")
  message(STATUS "Building oc_sort")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/algorithms/oc_sort")
  add_definitions(-DHAVE_OC_SORT)
endif()

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/algorithms/tracktrack")
  message(STATUS "Building tracktrack")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/algorithms/tracktrack")
  add_definitions(-DHAVE_TRACKTRACK)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/axtracker)

# Trackers library
file(GLOB TRACKERS_SOURCES "src/TrackerFactory.cpp" "src/cmc.cpp")

add_library(tracker_factory SHARED ${TRACKERS_SOURCES})
target_include_directories(
  tracker_factory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set RPATH to $ORIGIN so library can find co-located dependencies
set_target_properties(tracker_factory PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_RPATH "$ORIGIN"
)

target_link_libraries(
  tracker_factory PUBLIC axtracker ${OpenCV_LIBS}
)
target_link_options(tracker_factory PRIVATE -Wl,--exclude-libs,ALL)
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/algorithms/bytetrack")
  target_link_libraries(tracker_factory PUBLIC bytetrack)
endif()
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/algorithms/oc_sort")
  target_link_libraries(tracker_factory PUBLIC oc_sort)
endif()
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/algorithms/tracktrack")
  target_link_libraries(tracker_factory PUBLIC tracktrack)
endif()

# Example application (needs videoio/highgui, skip in minimal/CI builds)
option(BUILD_TRACKER_EXAMPLES "Build tracker example applications" ON)
if(BUILD_TRACKER_EXAMPLES)
  add_executable(track src/track.cpp)
  target_link_libraries(track PUBLIC tracker_factory)
endif()

# Install all targets
install(
  TARGETS tracker_factory
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
if(BUILD_TRACKER_EXAMPLES AND TARGET track)
  install(TARGETS track RUNTIME DESTINATION bin)
endif()

# Optionally install bytetrack and oc_sort targets if they are defined
if(TARGET bytetrack)
  install(TARGETS bytetrack LIBRARY DESTINATION lib)
endif()

if(TARGET oc_sort)
  install(TARGETS oc_sort LIBRARY DESTINATION lib)
endif()

if(TARGET axtracker)
  install(TARGETS axtracker LIBRARY DESTINATION lib)
endif()

if(TARGET tracktrack)
  install(TARGETS tracktrack LIBRARY DESTINATION lib)
endif()
