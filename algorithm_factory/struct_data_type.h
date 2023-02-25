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

//配置文件基类,自定义配置文件
struct ParmBase {
    // 1 从外部配置文件传入 =========================================
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

    // 2 代码运行过程中生成 =========================================
    // 推理输出结果结构:[batchSize,predictNums,predictLength]
    // 把所有输出拍平到一条直线时的数量,在onnx构建模型时就决定了
    int predictNums;
    // 每个预测的特征长度,对于目标检测来说,里面前5个预测特征通常是预测坐标和类别
    int predictLength;
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
    int classNums = 80;        //!> 检测类别数量
    float scoreThresh = 0.5;   //!> 得分阈值
    float iouThresh = 0.3;     //!> iou框阈值
};

// =============================================================================================
// 难以判断不同模型输出结果一定有什么,因此仅设一个空基类
struct ResultBase {
};

struct YoloFaceResult : public ResultBase {
};

// 将yolo 检测全部放在一个vector中输出
struct YoloDetectResult : public ResultBase {
//    每个检测框std::vector<float>存储, 一张图片所有检测框std::vector<std::vector<float>>, 多张图片结果结构就是下面的样子
    std::vector<std::vector<std::vector<float>>> result;
};

#ifdef __cplusplus
}
#endif


#endif //FACEFEATUREDETECTOR_REBUILD_STRUCT_FACE_FIELD_H