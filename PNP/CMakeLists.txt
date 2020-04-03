cmake_minimum_required(VERSION 2.8)
project(PNP)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Release")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV 3.2 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

set(SRC_CPP ${PROJECT_SOURCE_DIR}/src/main.cpp
            ${PROJECT_SOURCE_DIR}/src/PNP.cpp)

add_executable(PNP ${SRC_CPP})
target_link_libraries(PNP ${OpenCV_LIBS})