# OC-SORT tracking algorithm
# Parent handles: find_package(OpenCV/Eigen3), C++17 standard, install

add_library(oc_sort SHARED
  src/Association.cpp
  src/KalmanBoxTracker.cpp
  src/KalmanFilter.cpp
  src/lapjv.cpp
  src/OCSort.cpp
  src/Utilities.cpp
)

target_include_directories(oc_sort
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(oc_sort PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
target_link_options(oc_sort PRIVATE -Wl,--exclude-libs,ALL)

set_target_properties(oc_sort PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_RPATH "$ORIGIN"
)
