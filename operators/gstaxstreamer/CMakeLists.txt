cmake_minimum_required(VERSION 3.12)
project(axstreamer LANGUAGES C CXX VERSION 0.8)

find_package(PkgConfig)
pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GMODULE REQUIRED gmodule-2.0)
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_BASE REQUIRED gstreamer-base-1.0)
pkg_check_modules(GST_VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(GST_ALLOCATORS REQUIRED gstreamer-allocators-1.0)
pkg_check_modules(opencv4 REQUIRED opencv4)

include_directories(
    ${GLIB_INCLUDE_DIRS}
    ${GMODULE_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    ${GST_BASE_INCLUDE_DIRS}
    ${opencv4_INCLUDE_DIRS}
    ${GST_VIDEO_INCLUDE_DIRS}
    ${GST_ALLOCATORS_INCLUDE_DIRS}
)


set(SOURCE_FILES
    src/GstAlignedAllocator.cpp
    src/GstAxDataUtils.cpp
    src/GstAxDecode.cpp
    src/GstAxDistributor.cpp
    src/GstAxFunnel.cpp
    src/GstAxInference.cpp
    src/GstAxInferenceNet.cpp
    src/GstAxInPlace.cpp
    src/GstAxMeta.cpp
    src/GstAxPlugins.cpp
    src/GstAxStreamerUtils.cpp
    src/GstAxTransform.cpp
)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND SOURCE_FILES src/GstAxDmaBufAllocator.cpp)
else()
    list(APPEND SOURCE_FILES src/GstAxNoDmaBufAllocator.cpp)
endif()
add_library(gstaxstreamer SHARED ${SOURCE_FILES})

target_link_libraries(gstaxstreamer PUBLIC
    ${GLIB_LIBRARIES}
    ${GMODULE_LIBRARIES}
    ${GST_LIBRARIES}
    ${GST_BASE_LIBRARIES}
    ${GST_VIDEO_LIBRARIES}
    ${GST_ALLOCATORS_LIBRARIES}
    axstreamer
)

target_link_directories(gstaxstreamer PUBLIC
    ${GLIB_LIBRARY_DIRS}
    ${GMODULE_LIBRARY_DIRS}
    ${GST_LIBRARY_DIRS}
)

target_include_directories(gstaxstreamer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
