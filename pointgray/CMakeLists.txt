cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(flycapture CXX)

find_package(OpenCV REQUIRED)

link_directories("C:/Program Files/Point Grey Research/FlyCapture2/lib64")

add_executable(main main.cpp FlyCaptureWrapper.cpp)

target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(main PUBLIC "C:/Program Files/Point Grey Research/FlyCapture2/include")
target_link_libraries(main ${OpenCV_LIBS} FlyCapture2)
