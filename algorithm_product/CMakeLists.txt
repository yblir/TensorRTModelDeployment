
include_directories(
#        .
#        ./cuda
#        ../base
#        ../../../common/ini
#        ../../../common/status_code
#        ../../interface
        /usr/local/lib
        ../algorithm_factory
#        ../algorithm_product
        #    /usr/include
        #    /usr/local/include/
        #    /usr/local/include/opencv4/
        #    /usr/local/cuda-11.6/targets/x86_64-linux/include/
)

SET (CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -O0 -std=c++11 -g -ggdb -fPIC")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall -g -ggdb")


#link_directories(
#    /usr/local/cuda/lib64/
##        /usr/local/TensorRT-7.2.2.3/lib
##    /usr/lib64/
#   )
find_package(CUDA REQUIRED)
#CUDA_ADD_LIBRARY(yololayer STATIC ai_alg_trt_yolohead_layer_plugin.h ai_alg_trt_yolohead_layer_plugin.cu)


aux_source_directory(. DIR_LIB_SRCS)
ADD_LIBRARY(TrtFaceYolo SHARED ${DIR_LIB_SRCS} product.h)

TARGET_LINK_LIBRARIES(TrtFaceYolo
        -lcudart -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_cudaarithm
        -lopencv_cudawarping -lopencv_cudaimgproc
        )
# lnvinfer_plugin -lnvinfer  -lnvonnxparser
INSTALL(TARGETS TrtFaceYolo LIBRARY DESTINATION lib)
#INSTALL(FILES ../../interface/ai_img_alg_base.h DESTINATION include)
#INSTALL(FILES ai_alg_yoloface_if.h DESTINATION include)
#INSTALL(FILES ai_alg_yoloface_c_if.h DESTINATION include)
#INSTALL(FILES ai_alg_yoloface.h DESTINATION include)


#INSTALL(DIRECTORY model/ DESTINATION models/face_yolo_trt
#        PATTERN ".svn" EXCLUDE
#        )
#INSTALL(DIRECTORY etc/ DESTINATION etc)

#add_subdirectory(demo)
