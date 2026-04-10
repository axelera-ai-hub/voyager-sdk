# ByteTrack tracking algorithm
# Parent handles: find_package(OpenCV/Eigen3), C++17 standard, install

add_library(bytetrack SHARED
  src/BYTETracker.cpp
  src/kalmanFilter.cpp
  src/lapjv.cpp
  src/STrack.cpp
  src/utils.cpp
)

target_include_directories(bytetrack
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(bytetrack PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
target_link_options(bytetrack PRIVATE -Wl,--exclude-libs,ALL)

set_target_properties(bytetrack PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_RPATH "$ORIGIN"
)
