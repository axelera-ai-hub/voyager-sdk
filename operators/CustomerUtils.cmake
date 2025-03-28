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
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs TESTS)
    cmake_parse_arguments(CUSTOMER "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (GTEST_FOUND AND GMOCK_FOUND AND CUSTOMER_TESTS)
        set(test_target "unittest_${CUSTOMER_NAME}")
        add_executable(${test_target} ${CUSTOMER_TESTS})

        # Use the same settings as main project's tests
        target_include_directories(${test_target}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
            ${axstreamer_INCLUDE_DIRS}
            ${GMODULE_INCLUDE_DIRS}
            ${GSTREAMER_INCLUDE_DIRS}
            ${GTEST_INCLUDE_DIRS}
            ${GMOCK_INCLUDE_DIRS}
        )

        target_link_libraries(${test_target}
            PRIVATE
            ${GTEST_LIBRARIES}
            ${GMOCK_LIBRARIES}
            ${axstreamer_LIBRARIES}
            ${OpenCV_LIBS}
            ${OpenCL_LIBRARIES}
            ${GSTREAMER_LIBRARIES}
        )

        install(TARGETS ${test_target}
                DESTINATION ${unittest_install_dir})
        add_test(NAME ${test_target} COMMAND ${test_target})
    endif()
endfunction()
