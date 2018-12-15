# cmake -G"Visual Studio 15 2017 Win64" -DCMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE="x64" ..

cmake_minimum_required(VERSION 3.1)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDA REQUIRED)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd\\\"4819\\\"")
endif(MSVC)

cuda_add_executable(main main.cu)
