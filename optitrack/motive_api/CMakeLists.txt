cmake_minimum_required(VERSION 3.1)

project(MotiveAPITest CXX)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS} "C:/Program Files/OptiTrack/Motive/inc")

link_directories("C:/Program Files/OptiTrack/Motive/lib")

add_executable(main
    main.cpp
)

target_link_libraries(main NPTrackingToolsx64)
