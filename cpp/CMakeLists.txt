cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(lm-fusion VERSION 0.1)


include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)
set(CMAKE_VERBOSE_MAKEFILE on)

include(FetchContent)
include(ExternalProject)
set(FETCHCONTENT_QUIET off)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

if(NOT MSVC)
  # Keep the same with openfst, -fPIC or -fpic
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")
endif()

# third_party: gflags
# third_party: libtorch use FetchContent_Declare to download, and
# use find_package to find since libtorch is not a standard cmake project
set(PYTORCH_VERSION "1.11.0")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
  set(CMAKE_BUILD_TYPE "Release")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-${PYTORCH_VERSION}.zip")
else()
  message(FATAL_ERROR "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Windows', 'Linux' or 'Darwin')")
endif()
FetchContent_Declare(libtorch
  URL      ${LIBTORCH_URL}
  URL_HASH ${URL_HASH}
)
FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -DC10_USE_GLOG")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# utils
add_library(lm STATIC
  lm.cc
)
target_link_libraries(lm ${TORCH_LIBRARIES})

# add_executable(lm_main lm_main.cc)
# target_link_libraries(lm_main PUBLIC lm)

