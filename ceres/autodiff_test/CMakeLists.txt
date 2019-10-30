cmake_minimum_required(VERSION 3.1)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(main CXX)

find_package(Ceres REQUIRED)

add_executable(main main.cpp)
target_include_directories(main PUBLIC ${CERES_INCLUDE_DIRS})
target_link_libraries(main ${CERES_LIBRARIES})

