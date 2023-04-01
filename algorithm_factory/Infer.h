//
// Created by Administrator on 2023/3/2.
//

#ifndef TENSORRTMODELDEPLOYMENT_INFER_H
#define TENSORRTMODELDEPLOYMENT_INFER_H

#include <iostream>
#include <vector>
#include <future>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

using batchBoxesType = std::vector<std::vector<std::vector<float>>>;

//配置文件基类,自定义配置文件
struct ParamBase {
    // 1 从外部配置文件传入 ========================================================
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

    // 2 代码运行过程中生成 ========================================================
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

// 难以判断不同模型输出结果一定有什么,因此仅设一个空基类,唯一作用就是被product.h中productResult继承,实现多态效果
struct ResultBase {

};


class Infer {
public:
    Infer() = default;
    virtual ~Infer() = default;

//    virtual std::vector<std::shared_future<std::string>> commit(const std::string &imagePath) {};
    virtual std::shared_future<batchBoxesType> commit(const std::string &imagePath) {};

//    virtual std::vector<std::shared_future<std::string>> commit(const std::vector<std::string> &imagePaths) {};
    virtual std::shared_future<batchBoxesType> commit(const std::vector<std::string> &imagePaths) {};

    virtual int preProcess(ParamBase &param, cv::Mat &image, float *pinMemoryCurrentIn) = 0;
    virtual int postProcess(ParamBase &param, float *pinMemoryOut, int singleOutputSize, int outputNums, batchBoxesType &result) = 0;
};


std::shared_ptr<Infer> createInfer(ParamBase &param, const std::string &enginePath, Infer &curFunc);

typedef Infer *(*AlgorithmCreate)();

#endif //TENSORRTMODELDEPLOYMENT_INFER_H
