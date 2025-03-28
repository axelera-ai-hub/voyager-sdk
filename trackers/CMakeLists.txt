cmake_minimum_required(VERSION 3.10)
project(Trackers)

set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Python3 3.10 EXACT COMPONENTS Interpreter Development REQUIRED)

# Include directories for the entire project
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Add subdirectories
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/3rd_parties/bytetrack")
    message(STATUS "Building bytetrack")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rd_parties/bytetrack")
    add_definitions(-DHAVE_BYTETRACK)
endif()

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/3rd_parties/oc_sort")
    message(STATUS "Building oc_sort")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rd_parties/oc_sort")
    add_definitions(-DHAVE_OC_SORT)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/axtracker)

# Trackers library
file(GLOB TRACKERS_SOURCES "src/TrackerFactory.cpp")
add_library(tracker_factory SHARED ${TRACKERS_SOURCES})
target_include_directories(tracker_factory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tracker_factory PUBLIC axtracker ${OpenCV_LIBS} ${Python3_LIBRARIES})
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/3rd_parties/bytetrack")
    target_link_libraries(tracker_factory PUBLIC bytetrack)
endif()
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/3rd_parties/oc_sort")
    target_link_libraries(tracker_factory PUBLIC oc_sort)
endif()

# Example application
add_executable(track src/track.cpp)
target_link_libraries(track PUBLIC tracker_factory)

# Install all targets
install(TARGETS tracker_factory track
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# Optionally install bytetrack and oc_sort targets if they are defined
if(TARGET bytetrack)
    install(TARGETS bytetrack LIBRARY DESTINATION lib)
endif()

if(TARGET oc_sort)
    install(TARGETS oc_sort LIBRARY DESTINATION lib)
endif()

if(TARGET axtracker)
    install(TARGETS axtracker LIBRARY DESTINATION lib)
endif()
