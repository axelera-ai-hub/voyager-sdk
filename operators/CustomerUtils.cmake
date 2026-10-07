# Function to add customer libraries
function(add_customer_libraries CUSTOMER_NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SOURCES INCLUDE_DIRECTORIES)
    cmake_parse_arguments(CUSTOMER "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(ORIG_SOURCE_DIR ${CMAKE_SOURCE_DIR})
    set(CMAKE_SOURCE_DIR ${ORIG_SOURCE_DIR})

    foreach(source ${CUSTOMER_SOURCES})
        get_filename_component(target_name ${source} NAME_WE)
        create_shared_library(${target_name} ${source})

        # Record the library so the ctest runner can depend on it: customer tests
        # dlopen these .so files, so they must be built before ctest runs.
        set_property(GLOBAL APPEND PROPERTY AX_CUSTOMER_TEST_DEPS ${target_name})

        foreach(include_dir ${CUSTOMER_INCLUDE_DIRECTORIES})
            target_include_directories(${target_name}
                PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/${include_dir}
            )
        endforeach()
    endforeach()

    # Restore original source dir
    set(CMAKE_SOURCE_DIR ${ORIG_SOURCE_DIR})
endfunction()

# Function to add customer tests (if needed)
function(add_customer_tests CUSTOMER_NAME)
    cmake_parse_arguments(CUSTOMER "" "" "TESTS" ${ARGN})
    if (NOT CUSTOMER_TESTS)
        return()
    endif()

    # Customer subdirectories are configured before the main tree discovers
    # gtest/gmock, so discover them here under a private prefix. Using a private
    # prefix (rather than the shared GTEST_FOUND) keeps this self-contained and
    # avoids influencing discovery order elsewhere.
    find_package(PkgConfig QUIET)
    pkg_check_modules(AXCUSTOMER_GTEST gtest_main) # gtest + a default main()
    pkg_check_modules(AXCUSTOMER_GMOCK gmock)
    if (NOT (AXCUSTOMER_GTEST_FOUND AND AXCUSTOMER_GMOCK_FOUND))
        message(STATUS "gtest/gmock not found; skipping customer tests for ${CUSTOMER_NAME}")
        return()
    endif()

    set(test_target "unittest_${CUSTOMER_NAME}")
    add_executable(${test_target} ${CUSTOMER_TESTS})
    target_include_directories(${test_target}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests/ax           # shared test helpers (unittest_decode_common.h)
        ${CMAKE_SOURCE_DIR}/axstreamer/include
        ${GMODULE_INCLUDE_DIRS}
        ${AXCUSTOMER_GTEST_INCLUDE_DIRS}
        ${AXCUSTOMER_GMOCK_INCLUDE_DIRS}
    )
    # Link the discovered gtest/gmock (e.g. /usr/local/lib) rather than a stale
    # system copy that would mismatch the headers above. The axstreamer target
    # propagates its own include/link dependencies (OpenCV, gstreamer, ...).
    target_link_directories(${test_target}
        PRIVATE
        ${AXCUSTOMER_GTEST_LIBRARY_DIRS}
        ${AXCUSTOMER_GMOCK_LIBRARY_DIRS}
    )
    target_link_libraries(${test_target}
        PRIVATE
        axstreamer
        ${AXCUSTOMER_GTEST_LIBRARIES}
        ${AXCUSTOMER_GMOCK_LIBRARIES}
    )

    # Customer plugins are loaded by bare library name; point the loader at the
    # test's own build directory, where the customer library is co-located. The
    # compile definition lets the test set AX_SUBPLUGIN_PATH itself when run as a
    # bare binary; the ctest ENVIRONMENT covers runs under ctest.
    target_compile_definitions(${test_target}
        PRIVATE AX_TEST_PLUGIN_DIR="$<TARGET_FILE_DIR:${test_target}>")
    add_test(NAME ${test_target} COMMAND ${test_target})
    set_tests_properties(${test_target} PROPERTIES
        ENVIRONMENT "AX_SUBPLUGIN_PATH=$<TARGET_FILE_DIR:${test_target}>")

    # Ensure the test executable is built before the ctest runner executes it.
    set_property(GLOBAL APPEND PROPERTY AX_CUSTOMER_TEST_DEPS ${test_target})

    if (unittest_install_dir)
        install(TARGETS ${test_target} DESTINATION ${unittest_install_dir})
    endif()
endfunction()
