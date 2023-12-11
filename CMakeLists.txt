cmake_minimum_required(VERSION 3.26)
project(HW2)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(HW2 main.cpp)
