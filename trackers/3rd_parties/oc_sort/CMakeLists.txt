cmake_minimum_required(VERSION 3.10)
project(OC_SORT)

# Specify the required C++ standard
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} include)

# Source files
file(GLOB OC_SORT_SOURCES "src/*.cpp")

# Add OC_SORT library
add_library(oc_sort SHARED ${OC_SORT_SOURCES})

# Link libraries
target_link_libraries(oc_sort Eigen3::Eigen ${OpenCV_LIBS})
