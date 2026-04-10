# Voyager SDK Example showing use of AxInferenceNet with OpenCV
# Copyright Axelera AI, 2025
cmake_minimum_required(VERSION 3.12)
project(axinferencenet_example LANGUAGES CXX VERSION 0.8)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)

pkg_check_modules(axruntime REQUIRED axruntime)
pkg_check_modules(opencv4 REQUIRED opencv4)
pkg_check_modules(axstreamer REQUIRED axstreamer)

# Main example
add_executable(axinferencenet_example axinferencenet_example.cpp)

# Tensor output example
add_executable(axinferencenet_tensor axinferencenet_tensor.cpp)

# Tensor output using a locally defined custom plugin
add_executable(axinferencenet_local_plugin axinferencenet_local_plugin.cpp)

# Cascaded example
add_executable(axinferencenet_cascaded axinferencenet_cascaded.cpp)

# Tracker example
add_executable(axinferencenet_tracker axinferencenet_tracker.cpp)

foreach(target axinferencenet_example
               axinferencenet_tensor
               axinferencenet_local_plugin
               axinferencenet_cascaded
               axinferencenet_tracker)
    target_compile_options(${target} PRIVATE
        -Wall
        -Werror
        -Wno-sign-compare
        -Wno-unused-variable
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-enum-enum-conversion>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
    )
    target_include_directories(${target} PRIVATE
        ${axruntime_INCLUDE_DIRS}
        ${opencv4_INCLUDE_DIRS}
        ${axstreamer_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PUBLIC
        ${axruntime_LIBRARIES}
        ${opencv4_LIBRARIES}
        ${axstreamer_LIBRARIES}
    )
    target_link_directories(${target} PUBLIC
        ${axruntime_LIBRARY_DIRS}
        ${opencv4_LIBRARY_DIRS}
        ${axstreamer_LIBRARY_DIRS}
    )
    install(TARGETS ${target} RUNTIME DESTINATION bin)
endforeach()

# Only axinferencenet_tracker needs Eigen and tracker headers/libs
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

target_include_directories(axinferencenet_tracker PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/../trackers/include
)
target_link_libraries(axinferencenet_tracker PUBLIC
    ${CMAKE_SOURCE_DIR}/../operators/lib/libtracker_factory.so
)
