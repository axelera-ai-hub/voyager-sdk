# TrackTrack tracking algorithm
# Parent handles: find_package(OpenCV/Eigen3), C++17 standard, install

add_library(tracktrack SHARED
  src/Assignment.cpp
  src/Hungarian.cpp
  src/KalmanFilter.cpp
  src/Track.cpp
  src/TrackTrack.cpp
  # CMCComputer is used only by TrackTrack. Compiling it here rather than into
  # tracker_factory keeps the dependency one-way: tracker_factory already links
  # tracktrack, and MSVC resolves every symbol when it links a DLL, so the
  # reverse arrangement leaves tracktrack with an unresolved external.
  ../../src/cmc.cpp
)

target_include_directories(tracktrack
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${OpenCV_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..  # For cmc.hpp
)

target_link_libraries(tracktrack PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
target_link_options(tracktrack PRIVATE ${AX_TRACKER_LINK_OPTIONS})

set_target_properties(tracktrack PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_RPATH "$ORIGIN"
)
