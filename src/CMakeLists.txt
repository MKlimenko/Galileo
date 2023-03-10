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
	${CMAKE_CURRENT_LIST_DIR}/galileo/math.cpp
)

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${BIN_DST})
target_compile_options(${PROJECT_NAME} PUBLIC -fsycl-device-code-split=per_kernel)

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
)
