cmake_minimum_required(VERSION 2.8)
project(ICP)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Release")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV 3.2 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories("/usr/local/include/eigen3")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)
include_directories(${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})

set(SRC_CPP ${PROJECT_SOURCE_DIR}/src/main.cpp
            ${PROJECT_SOURCE_DIR}/src/ICP.cpp
            ${PROJECT_SOURCE_DIR}/src/ICP_G2O.cpp)

add_executable(ICP ${SRC_CPP})
target_link_libraries(ICP ${OpenCV_LIBS}
                          g2o_core
                          g2o_types_sba
                          g2o_stuff
                          ${CSPARSE_LIBRARY})