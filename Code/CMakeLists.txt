# declare the minimum version for CMake
cmake_minimum_required(VERSION 3.16)

# declare the project name
project(TensorLibrary LANGUAGES CXX)

# set the C++ standard
# set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP for parallel execution
find_package(OpenMP REQUIRED)

# Enable compiler warnings and advanced optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -O3             # Maximum optimization level for speed
        -march=native   # Enable CPU-specific SIMD instructions (AVX/AVX2)
    )
    
    # Link OpenMP flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# set the output directory for the compiled binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# enable testing
enable_testing()

# subdirectories
add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(bindings)
