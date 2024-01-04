//
// Created by Administrator on 2023/2/15.
//

#ifndef TENSORRT_PRO_PRODUCT_H
#define TENSORRT_PRO_PRODUCT_H

#include "../base_infer/infer.h"

// 以下是每个模型具体的参数配置, 每新加一个推理模型,都要在这里新增这个模型独特的参数,并继承通用参数配置BaseParam
//人脸检测配置
struct YoloFaceParam : public BaseParam {
    float scoreThresh = 0.5;   //!> 人脸框检测置信度阈值
    float iouThresh = 0.3;     //!> 人脸框IOU阈值
    bool useRefine = true;     //!> 是否需要图像旋转 true: 使用旋转优化检测 false: 不使用旋转正常检测
    std::shared_ptr<Infer> func;
};

// 检测相关
struct YoloDetectParam : public BaseParam {
    int classNums = 80;        //!> 检测类别数量
    float scoreThresh = 0.5;   //!> 得分阈值
    float iouThresh = 0.3;     //!> iou框阈值
    std::shared_ptr<Infer> trt;
//    // 将解析的算法配置也当做参数指针呢?
//    AlgorithmBase *trt = nullptr;
};

// ===============================================================================================

// ----------------------------------------------------------
// 以下三个结构体,从参数,函数到结果,必须一一对应                    |
// ----------------------------------------------------------

// todo 弃用, 越来越发现, productParam的存在很多余, 并没有减少多少代码量,反而增加了代码的可理解性
// 接受从外部传入的配置参数,并传递给算法
//struct productParam {
//    YoloFaceParam yoloFaceParam;
//    YoloDetectParam yoloDetectParam;
//
////    ~productParam() {
////        printf("productParam 执行析构\n");
////    };
//
//};

// todo 现在还不能把so解析处理的方法放到上面parm中. 如果放在子对象YoloDetectParm中,父类方法无法调用,需要强转类型,这样就不能
// todo 使用多态. 如果放在父类ParmBase中, 但ParmBase又在AlgorithmBase中调用,会出错.,解决办法,还是单独建一个结构体专门存储算法指针
// 从so动态库接受实现的算法函数调用,所有算法实现都共同基类AlgorithmBase
//struct productFunc {
//    AlgorithmBase *yoloFace;
//    AlgorithmBase *yoloDetect;
//};
//struct productFunc {
//    std::shared_ptr<Infer> trt;
//    std::shared_ptr<Infer> trt;
//
//    ~productFunc() {
//        printf("释放productFunc\n");
//    }
//
//
//};

//// 提取各个模型推理结果, 所有模型推理结果都可存储在三维vector.
//struct productResult : public ResultBase {
////  一张图片中每个预测项std::vector<float>存储, 一张图片所有预测项std::vector<std::vector<float>>, 最外层vector代表多张图片
//    // yolo通用检测结果
//    std::vector<std::vector<std::vector<float>>> detectResult;
//    // yolo人脸检测结果
//    std::vector<std::vector<std::vector<float>>> faceResult;
//
//};

// todo 从python代码中传入的参数, 根据实际需求传递定制化参数, 例如下面的classNums,iouThresh就是目标检测所需.
struct ManualParam {
//  todo fp16有时即使编译成功,也会推理时发生segment default错误.
//  todo 原因未知! 推测与可能与onnx文件有关, 难道要改变转onnx时参数的精度?
    bool fp32 = true;
//    bool fp16 = false;
    int classNums = 80;        //!> 检测类别数量
    float scoreThresh = 0.5;   //!> 得分阈值
    float iouThresh = 0.3;     //!> iou框阈值

    int gpuId = 0;
    int batchSize = 1;

    // 指定推理/模型输入高宽
    int inputHeight{};
    int inputWidth{};

    std::string onnxPath;
    std::string enginePath;

    // 推理时需要指定的输入输出节点名, 生成onnx文件时指定的输入输出名
    std::string inputName;
    std::string outputName;

//    char onnxPath[256]{};
//    char inputName[256]{};
//    char outputName[256]{};
//    char *onnxPath{};
//    char *inputName{};
//    char *outputName{};
};
#endif //TENSORRT_PRO_PRODUCT_H
