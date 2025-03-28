# Copyright Axelera AI, 2025
cmake_minimum_required(VERSION 3.12)
project(axsubplugins LANGUAGES C CXX VERSION 0.8)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ENABLE_CUSTOMER_COMPONENTS ON CACHE BOOL "Enable customer-specific components")
set(CMAKE_INSTALL_MESSAGE LAZY)

# Compiler-specific flags
add_compile_options(
    -Wall
    -Werror
    -Wno-sign-compare
    -Wno-unused-variable
)
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-enum-enum-conversion>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-anon-enum-enum-conversion>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type-c-linkage>
)
# add_compile_options(-fsanitize=address,undefined)
# add_link_options(-fsanitize=address,undefined)


find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCL OpenCL)
pkg_check_modules(opencv4 REQUIRED opencv4)
pkg_check_modules(nlohmann_json REQUIRED nlohmann_json)


add_subdirectory(PillowResize)
add_subdirectory(../trackers trackers)
add_subdirectory(axstreamer)
install(TARGETS axstreamer LIBRARY DESTINATION lib)
configure_file(${CMAKE_SOURCE_DIR}/axstreamer.pc.configure
               ${CMAKE_CURRENT_BINARY_DIR}/axstreamer.pc
               @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/axstreamer.pc
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig
)

pkg_check_modules(GMODULE gmodule-2.0)
pkg_check_modules(GLIB glib-2.0)
pkg_check_modules(GST gstreamer-1.0)
pkg_check_modules(GST_VIDEO gstreamer-video-1.0)
if(glib-2.0_FOUND AND glib-2.0_VERSION VERSION_GREATER_EQUAL "2.68.0")
    add_compile_definitions(GLIB_USE_G_MEMDUP2)
endif()
if (GMODULE_FOUND AND GLIB_FOUND AND GST_FOUND AND GST_VIDEO_FOUND)
    add_subdirectory(gstaxstreamer)
    install(TARGETS gstaxstreamer LIBRARY DESTINATION lib)
endif()

# Libraries
set(subplugin_prefix ${CMAKE_INSTALL_PREFIX})
set(subplugin_libdir ${subplugin_prefix}/${CMAKE_INSTALL_LIBDIR})
set(shared_libs PillowResize tracker_factory axstreamer)

function(create_shared_library target sources)
    add_library(${target} SHARED ${sources})
    set(shared_libs ${shared_libs} ${target} PARENT_SCOPE)
    target_include_directories(${target}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${opencv4_INCLUDE_DIRS}
        ${nlohmann_json_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/onnxruntime/onnxruntime/include
        axstreamer
    )
    target_link_libraries(${target}
        PRIVATE
        PillowResize
        tracker_factory
        ${opencv4_LIBRARIES}
        ${OpenCL_LIBRARIES}
        onnxruntime
        axstreamer
    )
    target_link_directories(${target}
        PRIVATE
        ${opencv4_LIBRARY_DIRS}
        ${CMAKE_SOURCE_DIR}/onnxruntime/onnxruntime/lib
        )

    if (APPLE)
        set_target_properties(
        ${target}
        PROPERTIES
        INSTALL_RPATH
        "${CMAKE_INSTALL_RPATH};${CMAKE_INSTALL_PREFIX}/lib;${subplugin_libdir}")
    endif()
    install(TARGETS ${target} DESTINATION ${subplugin_libdir})
endfunction()

create_shared_library(decode_classification src/AxDecodeClassification.cpp)
create_shared_library(decode_faceboxes src/AxDecodeFaceBoxes.cpp)
create_shared_library(decode_facerec src/AxDecodeFaceRec.cpp)
create_shared_library(decode_faciallandmarks src/AxDecodeLandmarks.cpp)
create_shared_library(decode_retinaface src/AxDecodeRetinaFace.cpp)
create_shared_library(decode_rtmdet src/AxDecodeRtmDet.cpp)
create_shared_library(decode_semantic_seg src/AxDecodeSemanticSeg.cpp)
create_shared_library(decode_ssd2 src/AxDecodeSsd2.cpp)
create_shared_library(decode_yolo "src/AxDecodeYolo.cpp;src/AxOnnxRuntimeHelper.cpp")
create_shared_library(decode_yolov5 src/AxDecodeYolo5.cpp)
create_shared_library(decode_yolov8 src/AxDecodeYolo8.cpp)
create_shared_library(decode_yolov8seg src/AxDecodeYolo8Seg.cpp)
create_shared_library(decode_yolox src/AxDecodeYoloX.cpp)
create_shared_library(inplace_addstreamid src/AxInPlaceAddStreamId.cpp)
create_shared_library(inplace_draw src/AxInPlaceDraw.cpp)
create_shared_library(inplace_filterdetections src/AxInPlaceFilterDetections.cpp)
create_shared_library(inplace_nms src/AxInPlaceNms.cpp)
create_shared_library(inplace_trackeraddclassifications src/AxInPlaceTrackerAddClassifications.cpp)
create_shared_library(inplace_trackeraddkeypoints src/AxInPlaceTrackerAddKeypoints.cpp)
create_shared_library(transform_centrecropextra src/AxTransformCentreCropExtra.cpp)
create_shared_library(transform_contrastnormalize src/AxTransformContrastNormalize.cpp)
create_shared_library(transform_cropresize src/AxTransformCropResize.cpp)
create_shared_library(transform_dequantize src/AxTransformDequantize.cpp)
create_shared_library(transform_facealign src/AxTransformFaceAlign.cpp)
create_shared_library(transform_padding src/AxTransformPadding.cpp)
create_shared_library(transform_resize src/AxTransformResize.cpp)
create_shared_library(transform_roicrop src/AxTransformRoiCrop.cpp)
create_shared_library(transform_totensor src/AxTransformToTensor.cpp)
create_shared_library(transform_yolopreproc src/AxTransformYoloPreProc.cpp)

if (OpenCL_FOUND)
    create_shared_library(transform_barrelcorrect_cl src/AxTransformBarrelCorrectCl.cpp)
    create_shared_library(transform_colorconvert src/AxTransformColorConvert.cpp)
    create_shared_library(transform_normalize_cl src/AxTransformNormalizeCl.cpp)
    create_shared_library(transform_perspective_cl src/AxTransformPerspectiveCl.cpp)
    create_shared_library(transform_resize_cl src/AxTransformResizeCl.cpp)
endif()

add_compile_options(-march=native)
create_shared_library(inplace_normalize src/AxInPlaceNormalize.cpp)
create_shared_library(transform_resizeratiocropexcess src/AxTransformResizeRatioCropExcess.cpp)
create_shared_library(inplace_tracker src/AxInPlaceTracker.cpp)




if(ENABLE_CUSTOMER_COMPONENTS)
    include(${CMAKE_SOURCE_DIR}/CustomerUtils.cmake)
    message(STATUS "Looking for customer components in: ${CMAKE_SOURCE_DIR}/../customers/*")

    file(GLOB CUSTOMER_DIRS "${CMAKE_SOURCE_DIR}/../customers/*")
    foreach(CUSTOMER_DIR ${CUSTOMER_DIRS})
        message(STATUS "Found customer directory: ${CUSTOMER_DIR}")
        if(EXISTS "${CUSTOMER_DIR}/CMakeLists.txt")
            get_filename_component(CUSTOMER_NAME ${CUSTOMER_DIR} NAME)
            message(STATUS "Including customer components: ${CUSTOMER_NAME}")
            add_subdirectory(${CUSTOMER_DIR} ${CMAKE_BINARY_DIR}/customers/${CUSTOMER_NAME})
        endif()
    endforeach()
endif()


pkg_check_modules(hailo_tappas hailo_tappas)
pkg_check_modules(gsthailometa gsthailometa)
if (GMODULE_FOUND AND GLIB_FOUND AND GST_FOUND AND GST_VIDEO_FOUND AND hailo_tappas_FOUND AND gsthailometa_FOUND)
    add_library(axmetatohailo SHARED src/GstAxMetaToHailo.cpp)
    target_include_directories(axmetatohailo
    SYSTEM PRIVATE
        ${tvm_runtime_INCLUDE_DIRS}
    )
    target_include_directories(axmetatohailo
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${axstreamer_INCLUDE_DIRS}
        ${GMODULE_INCLUDE_DIRS}
        ${GLIB_INCLUDE_DIRS}
        ${GST_INCLUDE_DIRS}
        ${hailo_tappas_INCLUDE_DIRS}
        ${gsthailometa_INCLUDE_DIRS}
    )
    target_link_libraries(axmetatohailo
        PRIVATE
        ${axstreamer_LIBRARIES}
        ${GMODULE_LIBRARIES}
        ${GLIB_LIBRARIES}
        ${GST_LIBRARIES}
        ${gsthailometa_LIBRARIES}
    )
    target_link_directories(axmetatohailo
        PRIVATE
        ${axstreamer_LIBRARY_DIRS}
        ${GMODULE_LIBRARY_DIRS}
        ${GLIB_LIBRARY_DIRS}
        ${GST_LIBRARY_DIRS}
        ${gsthailometa_LIBRARY_DIRS}
    )
    install(TARGETS axmetatohailo DESTINATION ${subplugin_libdir})
endif()

set(subplugin_bindir ${subplugin_prefix}/${CMAKE_INSTALL_BINDIR})
set(unittest_base_dir ${subplugin_bindir}/unittest-operators)
set(subplugin_prefix ${CMAKE_INSTALL_PREFIX})
set(path_ax_plugin_prefix ${CMAKE_BINARY_DIR})

add_definitions(-DAX_SUBPLUGIN_PATH=${path_ax_plugin_prefix})

include(CTest)

# Set unittest install dir
set(unittest_install_dir "${CMAKE_SOURCE_DIR}/tests")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GMODULE gmodule-2.0 REQUIRED)

# gtest
find_package(GTest)
pkg_check_modules(GMOCK gmock)
if (GMODULE_FOUND AND GLIB_FOUND AND GST_FOUND AND GST_VIDEO_FOUND AND GTEST_FOUND AND GMOCK_FOUND)
    # For now the unit tests are gstreamer dependendent. Its not strictly necessary
    # and we should add non-gstreamer tests in the future.
    find_package(OpenCV 4 REQUIRED)
    include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    find_package(OpenCL REQUIRED)

    if (OpenCL_FOUND)
    set(OPENCL_TEST_FILES
        tests/ax/unittest_transform_colorconvert.cc
        tests/ax/unittest_transform_normalize_cl.cc
        tests/ax/unittest_transform_barrelcorrect_cl.cc
        tests/ax/unittest_transform_perspective_cl.cc
        tests/ax/unittest_transform_resize_cl.cc
    )
    else()
        set(OPENCL_TEST_FILES "")
    endif()

    add_executable(unittest_ax
        tests/ax/unittest_ax_nms.cc
        tests/ax/unittest_ax_obj.cc
        tests/ax/unittest_ax_yolopreproc.cc
        tests/ax/unittest_ax.cc
        tests/ax/unittest_axinferencenet.cc
        tests/ax/unittest_axinference.cc
        tests/ax/unittest_decode_classification.cc
        tests/ax/unittest_decode_ssd2.cc
        tests/ax/unittest_decode_rtmdet.cc
        tests/ax/unittest_decode_semantic_seg.cc
        tests/ax/unittest_decode_yolo.cc
        tests/ax/unittest_decode_yolov5.cc
        tests/ax/unittest_decode_yolov8.cc
        tests/ax/unittest_decode_yolox.cc
        tests/ax/unittest_decode_yolov8seg.cc
        tests/ax/unittest_inplace_normalize.cc
        tests/ax/unittest_transform_centrecropextra.cc
        tests/ax/unittest_transform_dequantize.cc
        tests/ax/unittest_transform_padding.cc
        tests/ax/unittest_transform_resize.cc
        tests/ax/unittest_transform_resizeratiocropexcess.cc
        tests/ax/unittest_utils.cc
        ${OPENCL_TEST_FILES}
    )

    target_include_directories(unittest_ax
        PRIVATE
        ${GMODULE_INCLUDE_DIRS}
        ${GST_INCLUDE_DIRS}
    )

    target_link_directories(unittest_ax
        PUBLIC
        ${GST_LIBRARY_DIRS}
        ${GMOCK_LIBRARY_DIRS}
    )

    target_link_libraries(unittest_ax
        ${GTEST_LIBRARIES}
        ${GMOCK_LIBRARIES}
        ${OpenCV_LIBS}
        ${OpenCL_LIBRARIES}
        ${GST_LIBRARIES}
        axstreamer
        gstaxstreamer
        )

    install(TARGETS unittest_ax DESTINATION ${unittest_install_dir})

    add_test(NAME unittest_ax COMMAND unittest_ax)

    add_custom_target( run_unit_tests ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/RunUnitTests.touch
    )

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/RunUnitTests.touch
        COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_BINARY_DIR}/axstreamer:${CMAKE_CURRENT_BINARY_DIR}/gstaxstreamer:$ENV{LD_LIBRARY_PATH} ${CMAKE_CTEST_COMMAND} --output-on-failure
        COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/RunUnitTests.touch
        DEPENDS unittest_ax ${shared_libs}
    )
else()
    if (GMODULE_FOUND AND GLIB_FOUND AND GST_FOUND AND GST_VIDEO_FOUND AND GTEST_FOUND)
        message(WARNING "Unit tests will not be run. Please install libgtest-dev and libgmock-dev to enable unit tests.")
    else()
        message(WARNING "Unit tests will not be run. Currently they require gstreamer to be installed. Please install libgstreamer1.0-dev to enable unit tests.")
    endif()
endif()
