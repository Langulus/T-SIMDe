cmake_minimum_required(VERSION 3.20)
include(cmake/FetchLangulusModule.cmake)
project(Langulus_TSIMDe)

set(LANGULUS_ALIGNMENT 16 CACHE STRING "Overall langulus alignment")
add_definitions(-DLANGULUS_ALIGNMENT=${LANGULUS_ALIGNMENT})

# Check if this project is built as standalone, or a part of something else
if(PROJECT_IS_TOP_LEVEL)
    # We're building TSIMDe as standalone
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    fetch_langulus_module(Core)

    enable_testing()
	add_subdirectory(test)
endif()

# Configure SIMDe library
message(STATUS "Configuring external project: SIMDe...")
download_project(
	PROJ                SIMDe
	GIT_REPOSITORY      https://github.com/simd-everywhere/simde.git
	GIT_TAG             master
	GIT_PROGRESS        TRUE
	UPDATE_DISCONNECTED 1
)

# Build and install TSIMDe library
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SOURCE_FILES)
add_library(Langulus.TSIMDe INTERFACE)

target_link_libraries(Langulus.TSIMDe
	INTERFACE	Langulus.Core
)

target_include_directories(Langulus.TSIMDe
	INTERFACE	include/
	INTERFACE	${SIMDe_SOURCE_DIR}
)