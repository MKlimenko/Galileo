cmake_minimum_required (VERSION 3.24)

project (galileo_example)


set(CMAKE_CXX_STANDARD 20)
set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
set(BIN_DIR ${ROOT_DIR}/bin)
set(LIB_DIR ${ROOT_DIR}/lib)

if(PROJECT_IS_TOP_LEVEL)
	set(INC_DIR ${ROOT_DIR}/include)
else()
	set(INC_DIR ${ROOT_DIR}/src/galileo)
endif()

add_executable(
	${PROJECT_NAME}
	example.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${INC_DIR})
target_link_directories(${PROJECT_NAME} PUBLIC ${LIB_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC galileo)

set(ARCHIVE_OUTPUT_DIRECTORY ${BIN_DIR})
set(LIBRARY_OUTPUT_DIRECTORY ${BIN_DIR})
set(RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR})
set(EXECUTABLE_OUTPUT_PATH  ${BIN_DIR})

# add_custom_command (
# 	TARGET ${PROJECT_NAME}
# 	POST_BUILD
# 	COMMAND ${CMAKE_COMMAND} -E make_directory ${BIN_DIR}
# 	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${BIN_DIR}
# )
