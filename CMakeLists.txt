cmake_minimum_required(VERSION 3.15)
project(dma-image-processor VERSION 1.0.0 LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# Source files
set(SOURCES
    src/archiver.cpp
    src/metrics.cpp
    src/driver.cpp
    src/cnpy.cpp
    src/processor.cpp
    src/dma_source.cpp
)

# Find required libraries
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

# Find liburing (optional, for io_uring support)
find_library(URING_LIBRARY
    NAMES uring
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
    DOC "liburing library for io_uring support"
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    Threads::Threads
    ZLIB::ZLIB
)

# Link uring if available
if(URING_LIBRARY)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${URING_LIBRARY})
    target_compile_definitions(${PROJECT_NAME} PRIVATE PIPE_HAS_URING=1)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -mavx2
        -mfma
    )
endif()

# Optional: Add tests if needed
enable_testing()

# Optional: Add test executables
file(GLOB TEST_SOURCES
    tests/*.cpp
    tests/*/*.cpp
)

if(TEST_SOURCES)
    foreach(TEST_SOURCE ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
        
        # Determine which source files this test needs
        set(TEST_SOURCES_LIST ${TEST_SOURCE})
        if(TEST_NAME STREQUAL "test_dma_source")
            list(APPEND TEST_SOURCES_LIST src/dma_source.cpp src/cnpy.cpp)
        elseif(TEST_NAME STREQUAL "test_archiver")
            list(APPEND TEST_SOURCES_LIST src/archiver.cpp src/cnpy.cpp)
        elseif(TEST_NAME STREQUAL "test_processor")
            list(APPEND TEST_SOURCES_LIST src/processor.cpp src/cnpy.cpp)
        elseif(TEST_NAME STREQUAL "test_poisson")
            # poisson.hpp is header-only, no source file needed
        endif()
        add_executable(${TEST_NAME} ${TEST_SOURCES_LIST})
        
        target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
        target_link_libraries(${TEST_NAME}
            PRIVATE
            Threads::Threads
            ZLIB::ZLIB
        )
        # Add AVX2 flags for tests that use processor code
        if(TEST_NAME STREQUAL "test_processor")
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(${TEST_NAME} PRIVATE -mavx2 -mfma)
            endif()
        endif()
        # Link uring if available
        if(URING_LIBRARY)
            target_link_libraries(${TEST_NAME} PRIVATE ${URING_LIBRARY})
            target_compile_definitions(${TEST_NAME} PRIVATE PIPE_HAS_URING=1)
        endif()
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()

