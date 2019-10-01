cmake_minimum_required(VERSION 3.1)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cusolver_batch_solve CUDA)

find_package(CUDA REQUIRED)

add_executable(main_app main.cu)
target_link_libraries(main_app ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY})

