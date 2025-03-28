# Copyright Axelera AI, 2025
cmake_minimum_required(VERSION 3.12)
project(axstreamer LANGUAGES C CXX VERSION 0.8)

find_package(PkgConfig)
pkg_check_modules(opencv4 REQUIRED opencv4)
pkg_check_modules(OpenCL OpenCL)
pkg_check_modules(axruntime axruntime)

set(SOURCE_FILES
    src/AxInference.cpp
    src/AxInferenceAxRuntime.cpp
    src/AxInferenceNet.cpp
    src/AxLog.cpp
    src/AxNms.cpp
    src/AxOpenCl.cpp
    src/AxOpenClExtensions.cpp
    src/AxOpUtils.cpp
    src/AxStreamerUtils.cpp
)
add_library(axstreamer SHARED ${SOURCE_FILES})

target_include_directories(axstreamer PUBLIC
    ${OpenCL_INCLUDE_DIRS}
    ${opencv4_INCLUDE_DIRS}
    ${axruntime_INCLUDE_DIRS}/include
)

target_link_libraries(axstreamer PUBLIC
    ${opencv4_LIBRARIES}
    ${OpenCL_LIBRARIES}
    ${axruntime_LIBRARIES}
)

target_link_directories(axstreamer PUBLIC
    ${opencv4_LIBRARY_DIRS}
    ${OpenCL_LIBRARY_DIRS}
    ${axruntime_LIBRARY_DIRS}
)

target_include_directories(axstreamer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if (axruntime_FOUND)
    target_include_directories(axstreamer PRIVATE ${axruntime_INCLUDE_DIRS})
    target_link_libraries(axstreamer PUBLIC ${axruntime_LIBRARIES})
    target_link_directories(axstreamer PUBLIC ${axruntime_LIBRARY_DIRS})
    target_compile_definitions(axstreamer PUBLIC -DAXELERA_ENABLE_AXRUNTIME)
endif()
