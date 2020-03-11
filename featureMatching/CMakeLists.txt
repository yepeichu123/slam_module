cmake_minimum_required(VERSION 2.8) 
project(featureMatching)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Release")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(OpenCV 3.2 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SRC_CPP ${PROJECT_SOURCE_DIR}/src/FeatureMatching.cpp 
            ${PROJECT_SOURCE_DIR}/src/FeatureExtraction.cpp
            ${PROJECT_SOURCE_DIR}/src/main.cpp)

add_executable(featureMatching ${SRC_CPP})
target_link_libraries(featureMatching ${OpenCV_LIBS})