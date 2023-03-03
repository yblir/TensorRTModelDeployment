//
// Created by 12134 on 2023/2/9.
//
#ifndef INFERCODES_AlgorithmBase_H
#define INFERCODES_AlgorithmBase_H

// 所有待加速部署的模型,都继承工厂类,依据实际需求各自实现自己的代码
#include <iostream>
#include <istream>
//编译用的头文件
#include <NvInfer.h>
//onnx解释器头文件
#include <NvOnnxParser.h>
//推理用的运行时头文件
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <dlfcn.h>
#include <opencv2/opencv.hpp>

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
struct ResultBase{

};

// *****************************************************************************************************************************************

//通过智能指针管理nv, 内存自动释放,避免泄露.
template<typename T>
std::shared_ptr<T> prtFree(T *ptr) {
    return std::shared_ptr<T>(ptr, [](T *p) { delete p; });
}

inline const char *severity_string(nvinfer1::ILogger::Severity t);

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override;
};

class AlgorithmBase {
public:
    AlgorithmBase() = default;
    virtual ~AlgorithmBase() = default;

    /*
    *   @brief                  利用算法提供的配置结构初始化Ai基础算法，配置结构由具体算法各自定义
    *   @return                 成功返回0；失败返回对应错误码
    */
    virtual int initParam(void *param) = 0;

    // 图片预处理
    virtual int preProcess(ParamBase &parm, cv::Mat &image, float *pinMemoryCurrentIn) = 0;
    // 图片后处理
    virtual int postProcess(ParamBase &parm, float *pinMemoryOut, int singleOutputSize,
                            int outputNums, std::vector<std::vector<std::vector<float>>> &result) = 0;

//    virtual std::vector<std::vector<std::vector<float>>> postProcess(ParamBase &parm, float *pinMemoryOut, int singleOutputSize, int outputNums, ResultBase &result) = 0;
    // 推理内存中图片
//    virtual int inferImages(const std::vector<cv::Mat> &inputImages, outputBase &result) = 0;
    // 推理gpu中图片
//    virtual int inferGpuImages(const std::vector<cv::cuda::GpuMat> &inputImages, outputBase &result) = 0;

    // ================================================================================
    // 获得引擎名字, conf: 对应具体实现算法的结构体引用
    static std::string getEnginePath(const ParamBase &conf);
    //构建引擎文件,并保存到硬盘, 所有模型构建引擎文件方法都一样,如果加自定义层,继承算法各自实现
    static bool buildEngine(const std::string &onnxFilePath, const std::string &saveEnginePath, int maxBatch);
    //加载引擎到gpu,准备推理.
    static std::vector<unsigned char> loadEngine(const std::string &engineFilePath);
    // 创建推理engine
    static std::shared_ptr<nvinfer1::ICudaEngine> createEngine(const std::vector<unsigned char> &engineFile);
    //加载算法so文件
    static AlgorithmBase *loadDynamicLibrary(const std::string &soPath);

};

// 链接到动态库
typedef AlgorithmBase *(*AlgorithmCreate)();


#endif //INFERCODES_AlgorithmBase_H