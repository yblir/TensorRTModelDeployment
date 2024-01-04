# 写一份通用的cmake, 带()的是命令, 命令不区分大小写.
# 设置cmake需求的最小版本, 版本不符合抛出致命异常
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 工程名tensorRT_Pro, 支持语言是cuda和c++
project(deployment)

add_subdirectory(pybind11)

# -O0:编译时不优化代码 -fPIC:产生的代码中，没有绝对地址，全部使用相对地址
# std=c++11 -Wall: 编译后显示所有警告 -g: 使用gcc生成调试信息 -pthrad: 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -fPIC -pthread -g -std=c++11")
# 有Debug,Release两个选择. Release 有多一些优化?
#set(CMAKE_BUILD_TYPE "Debug")

# 然后在main函数前写上 #include “opencv/cv.h" 即可
# SYSTEM 指定目录当成系统的搜索目录 使用AFTER或BEFORE选项来指定是添加到列表的前面或者后面
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/TensorRT-8.6.1.6/include)
link_directories(/usr/local/TensorRT-8.6.1.6/lib)
include_directories(/usr/local/cuda/targets/x86_64-linux/include)
include_directories(/usr/local/include/opencv4)

# 添加需要链接的库文件路径，注意这里是全路径，要用在add_executable之前
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# 原始的CUDA_LIBRARIES=/usr/local/cuda/lib64/libcudart_static.adl/usr/lib/x86_64-linux-gnu/librt.s， 这是什么东西？
set(CUDA_LIBRARIES /usr/lib/wsl/lib/libcuda.so)
set(CUDART_LIBRARIES /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so)
set(i1 /usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvonnxparser.so)
set(i2 /usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvinfer.so)
set(i3 /usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib)

# ============================================================================================
# 对外封装时, 解除pybind11_add_module,target_link_libraries注释

pybind11_add_module(deployment
        interface/single_interface.cpp
        base_infer/single_infer.cpp

#        interface/thread_interface.cpp
#        base_infer/thread_infer.cpp

        product/YoloDetect.cpp
        utils/box_utils.cpp
        utils/general.cpp
        utils/loguru.cpp

        builder/trt_builder.cpp

        )
target_link_libraries(deployment PRIVATE pybind11::module
        ${OpenCV_LIBRARIES} ${pybind11_LIBRARIES}
        ${CUDA_LIBRARIES} ${CUDART_LIBRARIES}
        ${i1} ${i2}
        ${CMAKE_DL_LIBS}
        opencv_core opencv_imgcodecs opencv_imgproc opencv_videoio
        opencv_cudaarithm opencv_cudawarping opencv_cudaimgproc opencv_highgui
        )

# ===========================================================================================
# 调试时, 解除以下所有模块注释, 运行main_thread或main_single. 会爆出pybind11错误, 注释所有pybind11报错,调试完再解注释回来

#add_executable(run
##        main_single.cpp
##        interface/single_interface.cpp
##        base_infer/single_infer.cpp
#
#        main_thread.cpp
#        interface/thread_interface.cpp
#        base_infer/thread_infer.cpp
#
#        utils/loguru.cpp
#        utils/general.cpp
#
#        utils/box_utils.cpp
#        product/YoloDetect.cpp
#
#
#        base_infer/selfDataType.hpp
#        builder/trt_builder.cpp
#        )
#
## 为库或二进制可执行文件添加库链接，要用在add_executable之后,target是指通
## 过add_executable()和add_library()指令生成已经创建的目标文件run
#target_link_libraries(run ${DEP_LIBS} ${CUDA_LIBRARIES}
#        ${CUDART_LIBRARIES} ${OpenCV_LIBRARIES} ${i1} ${i2} ${i3} ${CMAKE_DL_LIBS}
#        opencv_core opencv_imgcodecs opencv_imgproc opencv_videoio
#        opencv_cudaarithm opencv_cudawarping opencv_cudaimgproc opencv_highgui
#        )
#link_libraries(${DEP_LIBS} ${CUDA_LIBRARIES}
#        ${CUDART_LIBRARIES} ${OpenCV_LIBRARIES} ${i1} ${i2} ${i3} ${CMAKE_DL_LIBS}
#        opencv_core opencv_imgcodecs opencv_imgproc opencv_videoio
#        opencv_cudaarithm opencv_cudawarping opencv_cudaimgproc opencv_highgui
#        )
#set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/dist)
#
#INSTALL(TARGETS run RUNTIME DESTINATION .)