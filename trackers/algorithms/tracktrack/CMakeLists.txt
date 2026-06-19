# TrackTrack tracking algorithm
# Parent handles: find_package(OpenCV/Eigen3), C++17 standard, install

add_library(tracktrack SHARED
  src/Assignment.cpp
  src/Hungarian.cpp
  src/KalmanFilter.cpp
  src/Track.cpp
  src/TrackTrack.cpp
)

target_include_directories(tracktrack
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${OpenCV_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..  # For cmc.hpp
)

target_link_libraries(tracktrack PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
target_link_options(tracktrack PRIVATE -Wl,--exclude-libs,ALL)

set_target_properties(tracktrack PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_RPATH "$ORIGIN"
)
