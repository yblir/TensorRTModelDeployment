//
// Created by Administrator on 2023/1/12.
//
#ifndef FACEFEATUREDETECTOR_REBUILD_STRUCT_FACE_FIELD_H
#define FACEFEATUREDETECTOR_REBUILD_STRUCT_FACE_FIELD_H

//编译用的头文件
//#include <NvInfer.h>
//onnx解释器头文件
//#include <NvOnnxParser.h>
//推理用的运行时头文件
#include <NvInferRuntime.h>
//#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif

//图片像素排列类型枚举类型
enum PixelFormat {
    // 暂时只支持B8G8R8像素排列的图片
    FAS_PF_RGB24_B8G8R8 = 0
};
// 单张方式输入图片信息接口基础结构，各算法可根据需要继承此结构扩展自己的输入结构
//struct InputDataBase {
//    cv::Mat image;
//};

//配置文件基类,自定义配置文件
struct ParmBase {
    // 1 从外部配置文件传入=========================================
    int gpuId = 0;
    std::string onnxPath;
    int batchSize = 1;
    bool useFp16 = false;
    // 推理时需要指定的输入输出节点名, 生成onnx文件时指定的输入输出名
    std::string inputName;
    std::string outputName;
    // 指定推理/模型输入宽高
    int inputHeight;
    int inputWidth;

    // 2 代码运行过程中生成=========================================
    // 存储一个batchSize的放射变换参数, 用于还原letterbox前的图片
    std::vector<std::vector<float>> d2is;
    std::string enginePath;
    // TensorRT 构建的引擎
    std::shared_ptr<nvinfer1::ICudaEngine> engine = nullptr;
    // 从engine生辰的上下文管理器
    std::shared_ptr<nvinfer1::IExecutionContext> context = nullptr;
};

//人脸检测配置
struct YoloFaceConfig : public ParmBase {
    float scoreThresh = 0.5;   //!> 人脸框检测置信度阈值
    float iouThresh = 0.3;     //!> 人脸框IOU阈值
    bool useRefine = true;     //!> 是否需要图像旋转 true: 使用旋转优化检测 false: 不使用旋转正常检测
};

struct YoloDetectConfig : public ParmBase {
    float scoreThresh = 0.5;   //!> 得分阈值
    float iouThresh = 0.3;     //!> iou框阈值
};

// =============================================================================================
// 难以判断不同模型输出结果一定有什么,因此仅设一个空基类
struct ResultBase {
//    int a=1;
};

struct YoloFaceResult : public ResultBase {
//    int b=1;
};

// 将yolo 检测全部放在一个vector中输出
struct YoloDetectResult : public ResultBase {
    std::vector<std::vector<std::vector<float>>> result;
};

#ifdef __cplusplus
}
#endif


#endif //FACEFEATUREDETECTOR_REBUILD_STRUCT_FACE_FIELD_H