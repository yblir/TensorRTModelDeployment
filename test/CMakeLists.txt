cmake_minimum_required(VERSION 3.10)
project(testCpp2)

#set("CMAKE_CXX_FLAGS -std=c++11 -lpthread -ldl")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -fPIC -pthread -g -std=c++11")
find_package(CUDA REQUIRED)
#find_package(OPENCV 4.0 COMPONENTS opencv_core opnecv_imgproc opencv_imgcodecs)
include_directories(/usr/local/include/opencv4)
#include_directories(./loguru.hpp)
#include_directories(/usr/local/cuda-11.6/targets/x86_64-linux/include)

#set(CUDA_LIB /usr/lib/wsl/lib/libcuda.so)
#set(CUDA_RT /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcudart.so)
add_executable(run main5.cpp infer.cpp infer.h)
#target_link_libraries(${PROJECT_NAME})
# /bin/mingw/lib/gcc/x86_64-w64-mingw32/11.2.0/libstdc++fs.a
#target_link_libraries(run ${CUDA_LIB} ${CUDA_RT})