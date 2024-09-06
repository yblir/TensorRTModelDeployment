//
// Created by Administrator on 2023/7/21.
//

#ifndef TENSORRTMODELDEPLOYMENT_SELFDATATYPE_HPP
#define TENSORRTMODELDEPLOYMENT_SELFDATATYPE_HPP

#include <vector>
#include <string>
#include <memory>

// tensorrt导入
#include <NvInfer.h>

#include <opencv2/opencv.hpp>
#include <future>
#include "../utils/loguru.hpp"

// ---------------------------------------------------------------------------------------------------------------------
// 宏定义PYBIND11表明是Python打包成调用模式，取消宏定义表示本地调试
// #define PYBIND11

// 仅thread_infer.cpp在中生效, 推理结果后处理时使用多线程处理
// #define POST_THREAD_POOL
// ---------------------------------------------------------------------------------------------------------------------

// 根据不同输出形式, 选择对应输出结果类型
//batch->image->box, [[[1,1,1,1],[2,2,2,2]],[[1,1,1,1]],...]
using batchBoxesType = std::vector<std::vector<std::vector<float>>>;

//每张图片上的框, 仅在thread_infer.cpp的后处理trtPost中有使用,std::map<int,imgBoxesType>
using imgBoxesType = std::vector<std::vector<float>>;

//输出[0.1, 0.3, 0.22, -0.5, ...]
//using batchBoxesType = std::vector<float>;

//输出[[0.3,0.1], [0.22,0.3], [-0.26,0.22], [-0.5,4] ...]
//using batchBoxesType = std::vector<std::vector<float>>;

using futureBoxes = std::shared_future<batchBoxesType>;

#define logInfo(...)    LOG_F(INFO, __VA_ARGS__)
#define logSuccess(...) LOG_F(SUCCESS, __VA_ARGS__)
#define logError(...)   LOG_F(ERROR, __VA_ARGS__)

enum class Mode : int {
    FP32,
    FP16
};

//配置文件基类,自定义配置文件
struct BaseParam {
    // 1 从外部配置文件传入 ========================================================
//    maxBatch指build trt引擎时的batch, 以后推理时batchSize不能大于maxBatch
    int maxBatch = 16;
    int gpuId = 0;
    std::string onnxPath;
    int batchSize = 1;

    // 使用fp32或fp16,当前仅支持两种选项,默认fp32
    Mode mode = Mode::FP32;

    // 推理时需要指定的输入输出节点名, 生成onnx文件时指定的输入输出名
    std::string inputName;
    std::string outputName;

    // 指定推理/模型输入高宽
    int inputHeight;
    int inputWidth;

    // 2 代码运行过程中生成 ========================================================
//    predictNums,predictLength就是trtOutputShape,可通过更通用的trtOutputShape实现
    // 推理输出结果结构:[batchSize,predictNums,predictLength]
//    // 把所有输出拍平到一条直线时的数量,在onnx构建模型时就决定了
//    int predictNums;
//    // 每个预测的特征长度,例如对于目标检测来说,里面前5个预测特征通常是预测坐标和类别
//    int predictLength;

    // 存储一个batchSize的仿射变换参数, 用于还原letterbox前的图片, 在预处理时传给job结构体, 通过队列最终传递给用于后处理还原原图坐标
    std::vector<std::vector<float>> preD2is;
//    用于后处理还原当前batch原图坐标, 不能一直用preD2is, 因为后一个batch的仿射参数会覆盖前一个的
    std::vector<std::vector<float>> postD2is;
    //当前正传处理图片的仿射变换参数, 每个线程独自创立这个参数, 不然多个预处理线程间会出现次序问题
//    float d2i[6];
    // 在代码运行时给出引擎文件路径,因为刚开始没有引擎文件
    std::string enginePath;

//    tensorrt推理时, engine指定的输入输出shape
    nvinfer1::Dims32 trtInputShape;
    nvinfer1::Dims32 trtOutputShape;

    // TensorRT 构建的引擎
    std::shared_ptr<nvinfer1::ICudaEngine> engine = nullptr;
    // 从engine生成的上下文管理器
    std::shared_ptr<nvinfer1::IExecutionContext> context = nullptr;
};

// commit输入数据类型必是以下中的一个
struct InputData {
//    std::vector<pybind11::array> pyMats;
    std::vector<cv::Mat> mats;
//    传入推理数据为单个或多个gpu图片矩阵,如果传入类型是以上两种类型,最后都要转化成GPU上
    std::vector<cv::cuda::GpuMat> gpuMats;
};


// 推理输入
struct Job {
//    float *inputTensor;
//    存储预处理节阶段的仿射变换参数,传递到后处理函数中,相当于一个中转站
    std::vector<std::vector<float>> d2is;
    int inferNum{};
};

// 推理输入图片路径+最终输出结果, 字段兼容各种输入类型. 不同输入类型对应传入对应字段中
struct futureJob {
    //取得是后处理后的结果
    std::shared_ptr<std::promise<batchBoxesType>> batchResult;

//    std::vector<pybind11::array> pyMats;
    std::vector<cv::Mat> mats;
    std::vector<cv::cuda::GpuMat> gpuMats;
};

// 推理输出
struct Out {
//    float *inferOut;
//    int index;
    std::vector<std::vector<float>> d2is;
    int inferNum{};
};

#endif //TENSORRTMODELDEPLOYMENT_SELFDATATYPE_HPP
