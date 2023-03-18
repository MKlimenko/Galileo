cmake_minimum_required (VERSION 3.24)

project(galileo_test)
find_package(IntelDPCPP REQUIRED)
add_executable (${PROJECT_NAME} 
							tests.cpp	
)
target_include_directories(${PROJECT_NAME} PRIVATE 
												${CMAKE_CURRENT_LIST_DIR}/../src/galileo
)
target_link_libraries(${PROJECT_NAME} PRIVATE 
												GTest::gtest 
												GTest::gtest_main 
												galileo
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

enable_testing()
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
set(ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../bin)
set(LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../bin)
set(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}/../bin)