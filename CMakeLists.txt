cmake_minimum_required(VERSION 3.20)
include(cmake/DownloadProject.cmake)

project(Langulus_TSIMDe)

# Check if this project is built as standalone, or a part of something else
if(PROJECT_IS_TOP_LEVEL)
    # We're building T-SIMDe as standalone
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    enable_testing()

    # Configure Langulus.Core
    message(STATUS "Configuring external project: Langulus.Core...")
    download_project(
	    PROJ                Langulus_Core
	    GIT_REPOSITORY      https://github.com/Langulus/Core.git
	    GIT_TAG             main
	    GIT_PROGRESS        TRUE
	    UPDATE_DISCONNECTED 1
    )
    include_directories(${Langulus_Core_SOURCE_DIR}/include)
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
include_directories(${SIMDe_SOURCE_DIR})

# Build and install T-SIMDe library
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SOURCE_FILES)
add_library(Langulus.TSIMDe INTERFACE)

target_include_directories(Langulus.TSIMDe
	INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Testing
add_subdirectory(test)
add_test(NAME Test.TSIMDe COMMAND Test.TSIMDe)
