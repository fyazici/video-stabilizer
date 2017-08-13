cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(cvtest)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options system)
find_package(glog REQUIRED)

add_executable(cvtest src/main.cpp)

target_link_libraries(cvtest m pthread ${OpenCV_LIBS} ${Boost_LIBRARIES} glog::glog)
