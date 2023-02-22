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
struct parmBase {
    float d2i[6];
    std::vector<std::vector<float>> d3i;
    int gpuId = 0;
    std::string onnxPath;
    std::string enginePath;
    int batchSize = 1;
    bool useFp16 = false;
    std::shared_ptr<nvinfer1::ICudaEngine> engine = nullptr;
    std::shared_ptr<nvinfer1::IExecutionContext> context = nullptr;
    // 推理时需要指定的输入输出节点名
    std::string inputName;
    std::string outputName;

    // 指定推理/模型输入宽高
    int inputHeight;
    int inputWidth;
};

//人脸检测配置
struct YoloFaceConfig : public parmBase {
    float scoreThresh = 0.5;   //!> 人脸框检测置信度阈值
    float iouThresh = 0.3;     //!> 人脸框IOU阈值
    bool useRefine = true;     //!> 是否需要图像旋转 true: 使用旋转优化检测 false: 不使用旋转正常检测
};

struct YoloDetectConfig : public parmBase {
    float scoreThresh = 0.5;   //!> 得分阈值
    float iouThresh = 0.3;     //!> iou框阈值
};

struct outputBase {

};

struct YoloFaceOutput : public outputBase {

};

struct TestSet {
    YoloFaceConfig yoloFace;
};
////人脸姿态评估配置
//struct FacePoseConfig {
//    //模型文件路径
//    char modelFile[256]{};
//    int batchSize = 16;         //!> batchSize设置
//    bool useFp16 = false;       //!> 是否使用fp16推理
//};

//人脸质量配置
//struct FaceQualityConfig {
//    //模型文件路径
//    char model_file[256]{};
//    int batch_size = 16;         //!> batchsize设置
//    bool use_fp16 = false;      //!> 是否使用fp16推理
//};

//人脸清晰度配置
//struct FaceSharpnessConfig {
//    //模型文件路径
//    char modelFile[256]{};
//    int batchSize = 16;        //!> batchsize设置
//    bool useFp16 = false;      //!> 是否使用fp16推理
//};

//struct FaceFeatureConfig {
//    //模型文件路径
//    char modelFile[256]{};
//
//    int batchSize = 16;     //!> batchsize设置
//    bool useFp16 = false;  //!> 是否使用fp16推理
//};

//人脸整体配置
//struct FaceTotalConfig {
//    int deviceId = 0;                //!> 统一配置推理gpu的设备id
//    int bitQualityType = 0x3f;        //!> 二进制位域形式表示需要抓取的人脸质量类型，例：0x1f表示取0,1,2,3,4这5种质量类型,-1非人脸使用最高位置1, 80000000
//    float score_sface_thresh = 0.9; //!> 单人脸模式阈值
//
//    //三种质量权重相加要等于1，否则提示参数异常
//    float blurWeight = 0.3f;          //!> 模糊质量权重
//    float illuminationWeight = 0.3f;  //!> 光照质量权重
//    float poseWeight = 0.4f;          //!> 姿态质量权重
//
//    struct RetinaFaceConfig retinaConf{};       // 人脸检测模块配置
//    struct FacePoseConfig poseConf{};         // 人脸姿态评估模块配置
//    struct FaceQualityConfig qualityConf{};      // 人脸检测质量评估模块配置
//    struct FaceSharpnessConfig sharpnessConf{};   // 人脸清晰度模块配置
//    struct FaceFeatureConfig featureConf{};      // 人脸特征提取模块配置
//};

//struct FaceResult {
//    float x1, y1, x2, y2;   // 人脸框四个角点坐标
//    float confidence;   // 人脸检测置信度
//    float angleP, angleR, angleY;   //姿态值: 俯仰角,滚转角,偏航角
//    float landmark[10]; // 人脸关键点坐标，10个值，对应左眼瞳孔，右眼瞳孔，鼻尖，左嘴角，右嘴角
//    float feature[256]; // 人脸特征值，256维。列表第一位为-2时表示不进行特征提取
//
//    // 人脸质量 (取值范围：0~4,-1，用于检查检测后的人脸为哪一类型)
//    /* ***************************************************
//    0: 正常(包括墨镜)
//    1 : 面部遮挡
//    2 : 姿态异常
//    3 : 低质人脸
//    4 : 色度异常
//    -1 : 未知
//    *************************************************** */
//    int qualityType;
//    float qualityScore;
//
//};

//// 人脸检测
//struct FaceDetectResult: public inferOutputBase {
//    //人脸框
//    cv::Rect2f faceRect;
//    //置信度
//    float confidence{};
//    //人脸姿态
//    float angleP{}, angleR{}, angleY{};
//    //人脸关键点
//    cv::Point2f landmark[5];
//    //人脸特征点
//    std::vector<float> feature;
//
//    //人脸质量结果及质量得分
//    int qualityType{};
//    float qualityScore{};
//
//};

//struct InferInputData: public inferInputBase{
//    int minFaceSize = 20;
//    int mode = 1;
//};
// =============================================================================

/* 定义单个人脸检测结果 *************************************/
//struct FaceOutput:public inferOutputBase{
//    float score;						//!> 置信度
//    cv::Rect2f rect;					//!> 框位置
//    std::vector<cv::Point2f> landmark;	//!> 特征点
//};

/* 定义人脸检测输入 *****************************************/
struct YoloFaceInput {
    cv::Mat imgMat;
    int minFaceSize = 20;        //!> 人脸最小过滤尺寸
    int mode = 1;                //!> 默认多人脸模式， 1： 多人脸 ， 0：单人脸
};
#ifdef __cplusplus
}
#endif


#endif //FACEFEATUREDETECTOR_REBUILD_STRUCT_FACE_FIELD_H