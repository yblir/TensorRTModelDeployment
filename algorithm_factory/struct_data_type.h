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



////人脸检测配置
//struct YoloFaceConfig : public ParmBase {
//    float scoreThresh = 0.5;   //!> 人脸框检测置信度阈值
//    float iouThresh = 0.3;     //!> 人脸框IOU阈值
//    bool useRefine = true;     //!> 是否需要图像旋转 true: 使用旋转优化检测 false: 不使用旋转正常检测
//};
//
//struct YoloDetectConfig : public ParmBase {
//    int classNums = 80;        //!> 检测类别数量
//    float scoreThresh = 0.5;   //!> 得分阈值
//    float iouThresh = 0.3;     //!> iou框阈值
//};

// =============================================================================================


//struct YoloFaceResult : public ResultBase {
//};
//
//// 将yolo 检测全部放在一个vector中输出
//struct YoloDetectResult : public ResultBase {
////    一张图片中每个预测项std::vector<float>存储, 一张图片所有检测框std::vector<std::vector<float>>, 最外层vector代表多张图片
//    std::vector<std::vector<std::vector<float>>> result;
//};

#ifdef __cplusplus
}
#endif


#endif //FACEFEATUREDETECTOR_REBUILD_STRUCT_FACE_FIELD_H