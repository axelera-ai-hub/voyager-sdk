cmake_minimum_required(VERSION 3.12)

# All files.cc in.and its subfolders.Define variable CURRENT_SRC
file(GLOB_RECURSE CURRENT_SRC "*.cc")

set(lib_name PillowResize)
add_library(${lib_name} "${CURRENT_SRC}")
set_target_properties(${lib_name} PROPERTIES VERSION ${PROJECT_VERSION})

target_include_directories(
  ${lib_name} PUBLIC $<INSTALL_INTERFACE:include> # build-tree
                     $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                     "${OpenCV_INCLUDE_DIRS}"

)# install-tree

# Define compile features for this target Define cpp "required" (_cxx_std) as
# standard
target_compile_features(${lib_name} PRIVATE cxx_std_17)
# Disable CXX extensions (it's better to do not rely upon compiler extensions)
set_target_properties(${lib_name} PROPERTIES CXX_EXTENSIONS OFF)

# Compile definitions (-D whatever)
target_compile_definitions(${lib_name} PUBLIC "")

if(ENABLE_CLANG_TIDY)
  set_target_properties(
    ${lib_name}
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               COMPILE_FLAGS "${WARNING_FLAGS}")

  if(CLANG_TIDY_EXE)
    set_target_properties(${lib_name} PROPERTIES CXX_CLANG_TIDY
                                                 "${DO_CLANG_TIDY}")
  endif()
endif()
