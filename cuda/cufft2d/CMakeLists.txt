cmake_minimum_required(VERSION 3.1)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(main CXX)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

link_directories(/usr/local/cuda-10.0/lib64)

cuda_add_executable(main main.cu)

target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main cufft culibos)
target_link_libraries(main ${OpenCV_LIBS})
