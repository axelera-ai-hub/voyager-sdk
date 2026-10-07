cmake_minimum_required(VERSION 3.10)
project(Trackers)

set(CMAKE_CXX_STANDARD 17)

# Find required packages
# Windows has no pkg-config ecosystem for OpenCV; vcpkg ships OpenCVConfig.cmake
# instead. This mirrors the lookup in axelera_runtime2/CMakeLists.txt.
if(WIN32)
  find_package(
    OpenCV CONFIG REQUIRED COMPONENTS core imgproc calib3d features2d video
  )
else()
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
  set(OpenCV_LIBS PkgConfig::OpenCV)
endif()
# No version floor, matching axelera_runtime2: Eigen's config declares
# same-major compatibility, so asking for 3.3 rejects the Eigen 5 that vcpkg
# ships. Both projects link Eigen3::Eigen into the same process and so must
# agree on which Eigen they found.
find_package(Eigen3 REQUIRED NO_MODULE)

if(MSVC)
  # _USE_MATH_DEFINES: MSVC does not define M_PI (used by oc_sort) without it.
  # /bigobj: the Eigen-heavy translation units exceed the default section limit.
  add_compile_definitions(_USE_MATH_DEFINES NOMINMAX WIN32_LEAN_AND_MEAN)
  add_compile_options(/bigobj)
  # MSVC exports nothing, and so emits no import library, unless symbols are
  # marked. Export everything, as axelera_runtime2 does for runtime2_core.
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  # Keep each library's statically linked dependencies out of its symbol table.
  set(AX_TRACKER_LINK_OPTIONS -Wl,--exclude-libs,ALL)
endif()

# Include directories for the entire project
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})

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
add_library(tracker_factory SHARED src/TrackerFactory.cpp)
target_include_directories(
  tracker_factory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set RPATH to $ORIGIN so library can find co-located dependencies
set_target_properties(tracker_factory PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_RPATH "$ORIGIN"
)

# Eigen matrices appear in TrackerFactory.h, so the dependency is public.
target_link_libraries(
  tracker_factory PUBLIC axtracker ${OpenCV_LIBS} Eigen3::Eigen
)
target_link_options(tracker_factory PRIVATE ${AX_TRACKER_LINK_OPTIONS})
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
# On Windows the DLL is the RUNTIME artifact and the import library the ARCHIVE
# one, so every library needs both destinations to be usable downstream.
foreach(lib IN ITEMS tracker_factory axtracker bytetrack oc_sort tracktrack)
  if(TARGET ${lib})
    install(
      TARGETS ${lib}
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
    )
  endif()
endforeach()

if(BUILD_TRACKER_EXAMPLES AND TARGET track)
  install(TARGETS track RUNTIME DESTINATION bin)
endif()
