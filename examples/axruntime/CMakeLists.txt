# Voyager SDK Example showing use of axruntime with OpenCV
# Copyright Axelera AI, 2025
cmake_minimum_required(VERSION 3.12)
project(axruntime_example LANGUAGES C CXX VERSION 0.8)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)

pkg_check_modules(axruntime REQUIRED axruntime)
pkg_check_modules(opencv4 REQUIRED opencv4)
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Werror
    -Wno-sign-compare
    -Wno-unused-variable
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-enum-enum-conversion>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${axruntime_INCLUDE_DIRS}/include
    ${opencv4_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${axruntime_LIBRARIES}
    ${opencv4_LIBRARIES}
)
target_link_directories(${PROJECT_NAME} PUBLIC
    ${axruntime_LIBRARY_DIRS}
    ${opencv4_LIBRARY_DIRS}
)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(PROGRAMS ${PROJECT_NAME}.py DESTINATION bin)
