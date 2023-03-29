cmake_minimum_required (VERSION 3.24)

project(galileo)

find_package(IntelDPCPP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
set(BIN_DST ${ROOT_DIR}/bin)
set(INC_DST ${ROOT_DIR}/include)
set(LIB_DST ${ROOT_DIR}/lib)

add_library (
	${PROJECT_NAME}
	SHARED
	${CMAKE_CURRENT_LIST_DIR}/galileo/common.hpp
	${CMAKE_CURRENT_LIST_DIR}/galileo/galileo.h
	${CMAKE_CURRENT_LIST_DIR}/galileo/galileo.def
	${CMAKE_CURRENT_LIST_DIR}/galileo/galileo.cpp
	${CMAKE_CURRENT_LIST_DIR}/galileo/unary.cpp
	${CMAKE_CURRENT_LIST_DIR}/galileo/unary.hpp
	${CMAKE_CURRENT_LIST_DIR}/galileo/binary.cpp
	${CMAKE_CURRENT_LIST_DIR}/galileo/binary.hpp
)

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${BIN_DST})
target_compile_options(${PROJECT_NAME} PUBLIC -fsycl-device-code-split=per_kernel)

if (WIN32)
	set(ONEAPI_LIBRARY_PATH $ENV{ONEAPI_ROOT}/compiler/latest/windows/bin)
	set(SYCL_LIBRARY ${ONEAPI_LIBRARY_PATH}/sycl6$<$<CONFIG:Debug>:d>.dll)
	set(OPENCL_PI ${ONEAPI_LIBRARY_PATH}/pi_opencl.dll)
	set(L0_PI ${ONEAPI_LIBRARY_PATH}/pi_level_zero.dll)

elseif (UNIX)
	set(ONEAPI_LIBRARY_PATH $ENV{ONEAPI_ROOT}/compiler/latest/linux/lib)
	set(SYCL_LIBRARY ${ONEAPI_LIBRARY_PATH}/libsycl.so*)
endif()

add_custom_command(
	TARGET ${PROJECT_NAME}
	POST_BUILD
	BYPRODUCTS ${INC_DST}/galileo.h
	COMMAND ${CMAKE_COMMAND} -E make_directory ${BIN_DST}
	COMMAND ${CMAKE_COMMAND} -E make_directory ${INC_DST}
	COMMAND ${CMAKE_COMMAND} -E make_directory ${LIB_DST}

	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${BIN_DST}
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_LIST_DIR}/galileo/galileo.h ${INC_DST}
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_LINKER_FILE:${PROJECT_NAME}> ${LIB_DST}
	
#	COMMAND ${CMAKE_COMMAND} -E copy ${SYCL_LIBRARY} ${BIN_DST}
#	COMMAND ${CMAKE_COMMAND} -E copy ${OPENCL_PI} ${BIN_DST}
#	COMMAND ${CMAKE_COMMAND} -E copy ${L0_PI} ${BIN_DST}
)

target_compile_options(${PROJECT_NAME} PUBLIC -ftime-trace)