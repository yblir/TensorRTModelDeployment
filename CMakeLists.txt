# 写一份通用的cmake, 带()的是命令, 命令不区分大小写.
# 设置cmake需求的最小版本, 版本不符合抛出致命异常
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
# 工程名tensorRT_Pro, 支持语言是cuda和c++
project(TensorRTModelDeployment)

set(CMAKE_CXX_STANDARD 17)
# -O0:编译时不优化代码 -fPIC:产生的代码中，没有绝对地址，全部使用相对地址
# std=c++11 -Wall: 编译后显示所有警告 -g: 使用gcc生成调试信息 -pthrad: 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -fPIC -pthread -g -std=c++11")
# 有Debug,Release两个选择. Release 有多一些优化?
set(CMAKE_BUILD_TYPE "Debug")

# QUIET,查找失败,不会提示,程序继续向下走. REQUIRED,一定要找到,找不到就停止
# COMPONENTS 一定要找到的组件,找不到就停止
# find_package(xxx ),会自动生成xxx_FOUND,xxx_DIR,xxx_INCLUDE_DIR,xxx_LIB(链接库路径)等变量
 find_package(CUDA REQUIRED)
 find_package(OpenCV 4.0 QUIET COMPONENTS core imgproc imgcodecs)

# 指定第三方库所在路径. 官方不建议使用了,建议用find_package()  find_library()
# 使用三方库三部曲: find_package,include_directories,target_link_libraries
# link_directories()

# 指定头文件的搜索路径, 去哪里找头文件
# 写上 include_directories(/usr/local/include) 来让库文件搜索以 /usr/local/include 为基础，
# 然后在main函数前写上 #include “opencv/cv.h" 即可
# SYSTEM 指定目录当成系统的搜索目录 使用AFTER或BEFORE选项来指定是添加到列表的前面或者后面
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/TensorRT-8.6.1.6/include)
link_directories(/usr/local/TensorRT-8.6.1.6/lib)
include_directories(/usr/local/cuda-12.1/targets/x86_64-linux/include)
include_directories(/usr/local/include/opencv4)

# 添加需要链接的库文件路径，注意这里是全路径，要用在add_executable之前
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
# 原始的CUDA_LIBRARIES=/usr/local/cuda/lib64/libcudart_static.adl/usr/lib/x86_64-linux-gnu/librt.s， 这是什么东西？
set(CUDA_LIBRARIES /usr/lib/wsl/lib/libcuda.so)
set(CUDART_LIBRARIES /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudart.so)
set(i1 /usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvonnxparser.so)
set(i2 /usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvinfer.so)

# 去哪里找库文件 .so/.lib/.ddl
link_directories()

add_subdirectory(product)

# 定义了这个工程会生成一个文件名为run 的可执行文件，相关的源文件是mian.cpp
add_executable(run
#        main.cpp
        main_thread.cpp

#        utils/general.h
        utils/general.cpp
#        utils/box_utils.h
        utils/box_utils.cpp
#        product/YoloFace.h product/YoloFace.cpp
#        product/YoloDetect.h product/YoloDetect.cpp
#        product/YoloDetect2.h
        product/YoloDetect.cpp
#        product/product.h
#
#        interface/face_interface.h interface/face_interface.cpp
#        interface/face_interface_thread.h
        interface/face_interface_thread.cpp

#        base_infer/infer.h
        base_infer/trt_infer.cpp
#        base_infer/factory.h base_infer/factory.cpp
        base_infer/selfDataType.hpp
        builder/trt_builder.cpp
#        builder/trt_builder.h
        )

# 为库或二进制可执行文件添加库链接，要用在add_executable之后,target是指通
# 过add_executable()和add_library()指令生成已经创建的目标文件run
target_link_libraries(run ${DEP_LIBS} ${CUDA_LIBRARIES}
        ${CUDART_LIBRARIES} ${OpenCV_LIBRARIES} ${i1} ${i2} ${i3} ${CMAKE_DL_LIBS}
        opencv_core opencv_imgcodecs opencv_imgproc opencv_videoio
        opencv_cudaarithm opencv_cudawarping opencv_cudaimgproc opencv_highgui
        )

set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/dist)

INSTALL(TARGETS run RUNTIME DESTINATION .)