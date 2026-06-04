# Option to enable sanitizer
option(ENABLE_SANITIZER "Enable address sanitizer" OFF)

# Copyright Axelera AI, 2025
cmake_minimum_required(VERSION 3.12)
project(axsubplugins LANGUAGES C CXX VERSION 0.8)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ENABLE_CUSTOMER_COMPONENTS ON CACHE BOOL "Enable customer-specific components")
set(CMAKE_INSTALL_MESSAGE NEVER)

# Compiler-specific flags
add_compile_options(
    -Wall
    -Werror
    # This is a long term project, huge number of errors
    -Wno-sign-compare
)
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-enum-enum-conversion>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-anon-enum-enum-conversion>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type-c-linkage>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-pessimizing-move>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-maybe-uninitialized>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=class-memaccess>
)
if(ENABLE_SANITIZER)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCL OpenCL)
pkg_check_modules(opencv4 REQUIRED opencv4)
pkg_check_modules(nlohmann_json REQUIRED nlohmann_json)
pkg_check_modules(libva libva)
if (OpenCL_FOUND)
    add_compile_definitions(HAVE_OPENCL)
endif()
if (libva_FOUND)
    add_compile_definitions(HAVE_LIBVA)
endif()

# Handle CUDA_AVAILABLE flag from Makefile
if(CUDA_AVAILABLE)
    add_compile_definitions(CUDA_AVAILABLE)
endif()

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
if(GLIB_FOUND AND GLIB_VERSION VERSION_GREATER_EQUAL "2.68.0")
    add_compile_definitions(GLIB_USE_G_MEMDUP2)
endif()

add_subdirectory(gstaxstreamer)
install(TARGETS gstaxstreamer LIBRARY DESTINATION lib)

# Libraries
set(subplugin_prefix ${CMAKE_INSTALL_PREFIX})
set(subplugin_bindir ${subplugin_prefix}/${CMAKE_INSTALL_BINDIR})
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
    if (LINUX)
      set_target_properties(
        ${target}
        PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
        INSTALL_RPATH
        "$ORIGIN"
      )
    endif()
    if (APPLE)
        set_target_properties(
        ${target}
        PROPERTIES
        INSTALL_RPATH
        "${CMAKE_INSTALL_RPATH};${CMAKE_INSTALL_PREFIX}/lib;${subplugin_libdir}")
    endif()
    install(TARGETS ${target} DESTINATION ${subplugin_libdir})
endfunction()

create_shared_library(trackerfilter_numsubtaskruns src/AxTrackerFilterNumSubtaskRuns.cpp)
create_shared_library(trackerattribute_bestscore src/AxTrackerAttributeBestScore.cpp)
create_shared_library(trackerattribute_mostfrequent src/AxTrackerAttributeMostFrequent.cpp)
create_shared_library(decode_classification src/AxDecodeClassification.cpp)
create_shared_library(decode_faceboxes src/AxDecodeFaceBoxes.cpp)
create_shared_library(decode_facerec src/AxDecodeFaceRec.cpp)
create_shared_library(decode_faciallandmarks src/AxDecodeLandmarks.cpp)
create_shared_library(decode_image src/AxDecodeImage.cpp)
create_shared_library(decode_ctc src/AxDecodeCTC.cpp)
create_shared_library(decode_retinaface src/AxDecodeRetinaFace.cpp)
create_shared_library(decode_rtmdet src/AxDecodeRtmDet.cpp)
create_shared_library(decode_semantic_seg src/AxDecodeSemanticSeg.cpp)
create_shared_library(decode_ssd2 src/AxDecodeSsd2.cpp)
create_shared_library(decode_to_raw_tensor src/AxDecodeToRawTensor.cpp)
create_shared_library(decode_yolo "src/AxDecodeYolo.cpp;src/AxOnnxRuntimeHelper.cpp")
create_shared_library(decode_yolov10_simple src/AxDecodeYolo10Simple.cpp)
create_shared_library(inplace_addstreamid src/AxInPlaceAddStreamId.cpp)
create_shared_library(inplace_addtiles src/AxInPlaceAddTiles.cpp)
create_shared_library(inplace_draw src/AxInPlaceDraw.cpp)
create_shared_library(inplace_filterdetections src/AxInPlaceFilterDetections.cpp)
create_shared_library(inplace_hidemeta src/AxInPlaceHideMeta.cpp)
create_shared_library(inplace_nms src/AxInPlaceNms.cpp)
create_shared_library(transform_centrecropextra src/AxTransformCentreCropExtra.cpp)
create_shared_library(transform_colorconvert src/AxTransformColorConvert.cpp)
create_shared_library(transform_contrastnormalize src/AxTransformContrastNormalize.cpp)
create_shared_library(transform_cropresize src/AxTransformCropResize.cpp)
create_shared_library(transform_dequantize src/AxTransformDequantize.cpp)
create_shared_library(transform_facealign "src/AxTransformFaceAlign.cpp;src/AxFaceAlignCommon.cpp")
create_shared_library(transform_padding src/AxTransformPadding.cpp)
create_shared_library(transform_paddingdequantize src/AxTransformPaddingDequantize.cpp)
create_shared_library(transform_resize src/AxTransformResize.cpp)
create_shared_library(transform_roicrop "src/AxTransformRoiCrop.cpp;src/AxRoiCropCommon.cpp")
create_shared_library(transform_totensor src/AxTransformToTensor.cpp)
create_shared_library(transform_postamble "src/AxTransformPostamble.cpp;src/AxOnnxRuntimeHelper.cpp")
create_shared_library(transform_yolopreproc src/AxTransformYoloPreProc.cpp)
create_shared_library(decode_multihead src/AxDecodeMultihead.cc)
create_shared_library(decode_embeddings src/AxDecodeEmbeddings.cc)
create_shared_library(decode_facenet src/AxDecodeFacenet.cc)
create_shared_library(transform_polar src/AxTransformPolar.cpp)
if (OpenCL_FOUND)
    create_shared_library(transform_barrelcorrect_cl src/AxTransformBarrelCorrectCl.cpp)
    create_shared_library(transform_colorconvert_cl src/AxTransformColorConvertCl.cpp)
    create_shared_library(transform_facealign_cl "src/AxTransformFaceAlignCl.cpp;src/AxFaceAlignCommon.cpp")
    create_shared_library(transform_normalize_cl src/AxTransformNormalizeCl.cpp)
    create_shared_library(transform_perspective_cl src/AxTransformPerspectiveCl.cpp)
    create_shared_library(transform_resize_cl src/AxTransformResizeCl.cpp)
    create_shared_library(transform_roicrop_cl "src/AxTransformRoiCropCl.cpp;src/AxRoiCropCommon.cpp")
    create_shared_library(transform_polar_cl src/AxTransformPolarCl.cpp)
endif()

add_compile_options(-march=native)
create_shared_library(inplace_normalize src/AxInPlaceNormalize.cpp)
create_shared_library(transform_resizeratiocropexcess src/AxTransformResizeRatioCropExcess.cpp)
create_shared_library(inplace_tracker src/AxInPlaceTracker.cpp)
create_shared_library(decode_yolov5 src/AxDecodeYolo5.cpp)
create_shared_library(decode_yolov8 src/AxDecodeYolo8.cpp)
create_shared_library(decode_yolox src/AxDecodeYoloX.cpp)





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
        ${libva_INCLUDE_DIRS}
    )
    target_link_libraries(axmetatohailo
        PRIVATE
        ${axstreamer_LIBRARIES}
        ${GMODULE_LIBRARIES}
        ${GLIB_LIBRARIES}
        ${GST_LIBRARIES}
        ${gsthailometa_LIBRARIES}
        ${libva_LIBRARIES}
    )
    target_link_directories(axmetatohailo
        PRIVATE
        ${axstreamer_LIBRARY_DIRS}
        ${GMODULE_LIBRARY_DIRS}
        ${GLIB_LIBRARY_DIRS}
        ${GST_LIBRARY_DIRS}
        ${gsthailometa_LIBRARY_DIRS}
        ${libva_LIBRARY_DIRS}
    )
    set_target_properties(axmetatohailo PROPERTIES INSTALL_RPATH "$ORIGIN")
    install(TARGETS axmetatohailo DESTINATION ${subplugin_libdir})
endif()

set(unittest_base_dir ${subplugin_bindir}/unittest-operators)
set(path_ax_plugin_prefix ${CMAKE_BINARY_DIR})

add_definitions(-DAX_SUBPLUGIN_PATH=${path_ax_plugin_prefix})

if(NOT ENABLE_SANITIZER)
    include(CTest)
endif()

# Set unittest install dir
set(unittest_install_dir "${CMAKE_SOURCE_DIR}/tests")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GMODULE gmodule-2.0 REQUIRED)

# gtest/gmock via pkg-config to avoid picking up Conda-provided CMake packages.
pkg_check_modules(GTEST gtest)
pkg_check_modules(GMOCK gmock)
if (GMODULE_FOUND AND GLIB_FOUND AND GST_FOUND AND GST_VIDEO_FOUND AND GTEST_FOUND AND GMOCK_FOUND)
    # For now the unit tests are gstreamer dependendent. Its not strictly necessary
    # and we should add non-gstreamer tests in the future.
    find_package(OpenCV 4 REQUIRED)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )

    find_package(OpenCL REQUIRED)

    if (OpenCL_FOUND)
    set(OPENCL_TEST_FILES
        tests/ax/unittest_opencl.cc
        tests/ax/unittest_transform_colorconvert_cl.cc
        tests/ax/unittest_transform_normalize_cl.cc
        tests/ax/unittest_transform_barrelcorrect_cl.cc
        tests/ax/unittest_transform_perspective_cl.cc
        tests/ax/unittest_transform_resize_cl.cc
        tests/ax/unittest_transform_roicrop_cl.cc
        tests/ax/unittest_transform_polar_cl.cc
    )
    else()
        set(OPENCL_TEST_FILES "")
    endif()

    add_executable(unittest_ax
        tests/ax/unittest_ax_filterdetections.cc
        tests/ax/unittest_ax_meta.cc
        tests/ax/unittest_ax_nms.cc
        tests/ax/unittest_ax_obj.cc
        tests/ax/unittest_ax_yolopreproc.cc
        tests/ax/unittest_ax.cc
        tests/ax/unittest_axinferencenet.cc
        tests/ax/unittest_decode_classification.cc
        tests/ax/unittest_decode_ssd2.cc
        tests/ax/unittest_decode_rtmdet.cc
        tests/ax/unittest_decode_semantic_seg.cc
        tests/ax/unittest_decode_yolo.cc
        tests/ax/unittest_decode_yolov5.cc
        tests/ax/unittest_decode_yolov8.cc
        tests/ax/unittest_decode_yolo10simple.cc
        tests/ax/unittest_decode_yolox.cc
        tests/ax/unittest_decode_yolov8seg.cc
        tests/ax/unittest_decode_image.cc
        tests/ax/unittest_decode_to_raw_tensor.cc
        tests/ax/unittest_decode_ctc.cc
        tests/ax/unittest_decode_facenet.cc
        tests/ax/unittest_inplace_normalize.cc
        tests/ax/unittest_inplace_addtiles.cc
        tests/ax/unittest_transform_centrecropextra.cc
        tests/ax/unittest_transform_colorconvert.cc
        tests/ax/unittest_transform_dequantize.cc
        tests/ax/unittest_transform_padding.cc
        tests/ax/unittest_transform_paddingdequantize.cc
        tests/ax/unittest_transform_postamble.cc
        tests/ax/unittest_transform_resize.cc
        tests/ax/unittest_transform_resizeratiocropexcess.cc
        tests/ax/unittest_transform_facealign.cc
        tests/ax/unittest_transform_roicrop.cc
        tests/ax/unittest_utils.cc
        ${OPENCL_TEST_FILES}
    )

    target_include_directories(unittest_ax
        PRIVATE
        ${GMODULE_INCLUDE_DIRS}
        ${nlohmann_json_INCLUDE_DIRS}
        ${GST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/onnxruntime/onnxruntime/include
    )

    target_link_directories(unittest_ax
        PUBLIC
        ${GST_LIBRARY_DIRS}
        ${GMOCK_LIBRARY_DIRS}
        ${libva_LIBRARY_DIRS}
        ${CMAKE_SOURCE_DIR}/onnxruntime/onnxruntime/lib
    )

    target_link_libraries(unittest_ax
        ${GTEST_LIBRARIES}
        ${GMOCK_LIBRARIES}
        ${OpenCV_LIBS}
        ${OpenCL_LIBRARIES}
        ${GST_LIBRARIES}
        Eigen3::Eigen
        axstreamer
        gstaxstreamer
        onnxruntime
        )

    install(TARGETS unittest_ax DESTINATION ${unittest_install_dir})

    add_test(NAME unittest_ax COMMAND unittest_ax)

    if (NOT ENABLE_SANITIZER)
        add_custom_target( run_unit_tests ALL
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/RunUnitTests.touch
        )
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/RunUnitTests.touch
            COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_BINARY_DIR}/axstreamer:${CMAKE_CURRENT_BINARY_DIR}/gstaxstreamer:$ENV{LD_LIBRARY_PATH} ${CMAKE_CTEST_COMMAND} --output-on-failure
            COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/RunUnitTests.touch
            DEPENDS unittest_ax ${shared_libs}
        )
    endif()

    # Create assets directory in build directory
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/assets)

    # Copy ONNX file to assets directory
    configure_file(
        ${CMAKE_SOURCE_DIR}/tests/ax/assets/add_one.onnx
        ${CMAKE_BINARY_DIR}/assets/add_one.onnx
        COPYONLY
    )

    configure_file(
        ${CMAKE_SOURCE_DIR}/tests/ax/assets/add_tensors.onnx
        ${CMAKE_BINARY_DIR}/assets/add_tensors.onnx
        COPYONLY
    )

    # Make sure the test has access to the ONNX file
    target_compile_definitions(unittest_ax PRIVATE
        TEST_ASSETS_DIR="${CMAKE_BINARY_DIR}/assets"
    )

else()
    if (GMODULE_FOUND AND GLIB_FOUND AND GST_FOUND AND GST_VIDEO_FOUND)
        message(WARNING "Unit tests will not be run. Please install libgtest-dev and libgmock-dev to enable unit tests.")
    else()
        message(WARNING "Unit tests will not be run. Currently they require gstreamer to be installed. Please install libgstreamer1.0-dev to enable unit tests.")
    endif()
endif()
