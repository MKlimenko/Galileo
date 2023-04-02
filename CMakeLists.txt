cmake_minimum_required (VERSION 3.24)

find_program(INTEL_DPCPP_COMPILER "icx")
if (NOT INTEL_DPCPP_COMPILER)   
  message(FATAL_ERROR "Please set up an environment first by calling source {ONEAPI_ROOT}/setvars.sh on Linux or {ONEAPI_ROOT}/setvars.bat on Windows" )
endif()

# set compiler to avoid exporting environemnt variables
if (WIN32)
	set(CMAKE_CXX_COMPILER "icx")
elseif (UNIX)
	set(CMAKE_CXX_COMPILER "icpx")
endif()

set(GALILEO_VER_MAJOR 0)
set(GALILEO_VER_MINOR 0)
set(GALILEO_VER_PATCH 1)
add_definitions(-DGALILEO_VERSION_MAJOR=${GALILEO_VER_MAJOR} -DGALILEO_VERSION_MINOR=${GALILEO_VER_MINOR} -DGALILEO_VERSION_PATCH=${GALILEO_VER_PATCH})

project (galileo_root)

add_subdirectory(src)

add_subdirectory(examples)
add_dependencies(galileo_example galileo)

find_package(GTest CONFIG)
if (GTest_FOUND)
	enable_testing()
	add_subdirectory ("test")
	add_dependencies(galileo_test galileo)
endif()

find_package(benchmark CONFIG)
if (benchmark_FOUND)
	add_subdirectory ("benchmark")
	add_dependencies(galileo_benchmark galileo)
endif()
