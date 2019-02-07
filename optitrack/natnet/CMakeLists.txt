cmake_minimum_required(VERSION 3.1)

project(NatNetTest CXX)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})
link_directories("C:/Program Files/OptiTrack/NatNetSDK/lib/x64")

add_executable(main
    main.cpp
    optitrack_client.cpp
)

target_include_directories(main PUBLIC "C:/Program Files/OptiTrack/NatNetSDK/include")
target_link_libraries(main NatNetLib)
