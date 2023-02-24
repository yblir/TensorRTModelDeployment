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

#include "struct_data_type.h"

//通过智能指针管理nv, 内存自动释放,避免泄露.
template<typename T>
std::shared_ptr<T> prtFree(T *ptr) {
    return std::shared_ptr<T>(ptr, [](T *p) { delete p; });
}

inline const char *severity_string(nvinfer1::ILogger::Severity t) {
    switch (t) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:
            return "error";
        case nvinfer1::ILogger::Severity::kWARNING:
            return "warning";
        case nvinfer1::ILogger::Severity::kINFO:
            return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE:
            return "verbose";
        default:
            return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if (severity == Severity::kWARNING) {
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            } else if (severity <= Severity::kERROR) {
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            } else {
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
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

    // 图片预处理 todo 为何不能传引用
    virtual int preProcess(ParmBase &parm, cv::Mat &image, float *pinMemoryCurrentIn) = 0;
    // 图片后处理
    virtual int postProcess(ParmBase &parm, std::vector<cv::Mat> &images,
                            float *pinMemoryOut, int singleOutputSize, ResultBase &result) = 0;
    // 推理内存中图片
//    virtual int inferImages(const std::vector<cv::Mat> &inputImages, outputBase &result) = 0;
    // 推理gpu中图片
//    virtual int inferGpuImages(const std::vector<cv::cuda::GpuMat> &inputImages, outputBase &result) = 0;

    // ================================================================================
    // 获得引擎名字, conf: 对应具体实现算法的结构体引用
    static std::string getEnginePath(const ParmBase &conf);
    //构建引擎文件,并保存到硬盘, 所有模型构建引擎文件方法都一样,如果加自定义层,继承算法各自实现
    static bool buildEngine(const std::string &onnxFilePath, const std::string &saveEnginePath, int maxBatch);
    //加载引擎到gpu,准备推理.
    static std::vector<unsigned char> loadEngine(const std::string &engineFilePath);
    // 创建推理engine
    static std::shared_ptr<nvinfer1::ICudaEngine> createEngine(const std::vector<unsigned char> &engineFile);
    //加载算法so文件
    static AlgorithmBase *loadDynamicLibrary(const std::string &soPath);

};

typedef AlgorithmBase *(*AlgorithmCreate)();


#endif //INFERCODES_AlgorithmBase_H