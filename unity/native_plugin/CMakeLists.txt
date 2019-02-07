cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(nativeplugintest CXX)

if (APPLE)
add_library(mylib MODULE mylib.cpp)
set_target_properties(mylib PROPERTIES BUNDLE TRUE)
else()
add_library(mylib SHARED mylib.cpp)
endif()

