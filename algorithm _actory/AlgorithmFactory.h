//
// Created by 12134 on 2023/2/9.
//
#ifndef INFERCODES_ALGORITHMFACTORY_H
#define INFERCODES_ALGORITHMFACTORY_H

#endif //INFERCODES_ALGORITHMFACTORY_H
// 所有待加速部署的模型,都继承工厂类,依据实际需求各自实现自己的代码
#include <istream>
//编译用的头文件
#include <NvInfer.h>
//onnx解释器头文件
#include <NvOnnxParser.h>
//推理用的运行时头文件
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

class AlgorithmFactory {
public:
    AlgorithmFactory() = default;
    virtual ~AlgorithmFactory() = default;
    /*
    *   @brief                  按照给定配置文件初始化Ai基础算法
    *   @param cfg_file         ai基础算法所需要的配置文件
    *   @return                 成功返回0；失败返回对应错误码
    */
    virtual int FAImgAlgInit(std::string& cfg_file) = 0;

    /*
    *   @brief                  利用算法提供的配置结构初始化Ai基础算法，配置结构由具体算法各自定义
    *   @return                 成功返回0；失败返回对应错误码
    */
    virtual int FAImgAlgInit(void* param) = 0;

    /*
    *   @brief                  图像算法推理（普通模式）
    *   @param frame            单帧图片信息（Mat形式输入）
    *   @param result           ai算法推理输出的结果集合，需要经过ai识别功能模块中定义的输出类型转换，内存需要用户释放
    *   @param res_count        ai算法推理输出的结果数
    *   @return                 成功返回0；失败返回对应错误码
    */
    virtual int FAImgAlgInfer(const FAImgAlgInputBase_t* frame, FAImgAlgOutputBase_t* result) = 0;
    //构建引擎文件,并保存到硬盘
    bool buildEngine(const std::string & onnxFilePath,const std::string &saveEnginePath);
    //加载引擎到gpu,准备推理.
    bool loadEngine(const std::string &engineFilePath);
private:
    void *m_pfactory;
};