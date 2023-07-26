include_directories(
        /usr/local/lib
        ../base_infer
        ../interface
        ../builder
)

SET (CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -O0 -std=c++11 -g -ggdb -fPIC")

find_package(CUDA REQUIRED)

aux_source_directory(. DIR_LIB_SRCS)
#ADD_LIBRARY(TrtFaceYolo SHARED ${DIR_LIB_SRCS} product.h YoloDetect.cpp YoloDetect.h)
ADD_LIBRARY(TrtYoloDetect SHARED ${DIR_LIB_SRCS}
        product.h
        YoloDetect.cpp YoloDetect.h
#        YoloFace.h YoloFace.cpp
        ../utils/general.h ../utils/general.cpp ../utils/box_utils.h ../utils/box_utils.cpp
        ../base_infer/selfDataType.hpp
        ../base_infer/infer.h ../base_infer/trt_infer.cpp
        ../interface/face_interface.h ../interface/face_interface_thread.cpp
        ../builder/trt_builder.cpp ../builder/trt_builder.h
        )
#TARGET_LINK_LIBRARIES(TrtFaceYolo
#        -lcudart -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_cudaarithm
#        -lopencv_cudawarping -lopencv_cudaimgproc
#        )
TARGET_LINK_LIBRARIES(TrtYoloDetect
        -lcudart -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_cudaarithm
        -lopencv_cudawarping -lopencv_cudaimgproc
        )
# lnvinfer_plugin -lnvinfer  -lnvonnxparser
#INSTALL(TARGETS TrtFaceYolo LIBRARY DESTINATION lib)
INSTALL(TARGETS TrtYoloDetect LIBRARY DESTINATION lib)
