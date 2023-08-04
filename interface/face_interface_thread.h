//
// Created by Administrator on 2023/1/9.
//
#ifndef FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H
#define FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H
#include <tuple>

#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
//#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaarithm.hpp"

#include "../product/product.h"
#include "../base_infer/infer.h"
#include "../product/YoloDetect.h"
//#define checkRuntime(op) check_cuda_runtime((op),#op,__FILE__,__LINE__)

using Handle = void *;

// 定义全局变量
//extern productParam param;
//bool check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line);

// 加{},说明创建的对象为nullptr, 存储从动态库解析出来的算法函数和类
// 初始化过程中,各个模型都会用到的通用步骤
//int initCommon(BaseParam &confSpecific, class AlgorithmBase *funcSpecific);

// 测试重构一下?
//int inferEngine(productParam &param, productFunc &func, std::vector<cv::Mat> &images, int &res_num);
//int initEngine(productParam &param, productFunc &func);
//extern "C" int initEngine(productParam &param);
extern "C" int initEngine(Handle &engine, externalParam &inputParam);
//extern "C" int initEngine(externalParam &param);
//int inferEngine(productParam &param, productFunc &func, std::vector<cv::Mat> &mats, productResult &out);
//int inferEngine(productParam &param, productFunc &func, std::vector<cv::cuda::GpuMat> &matVector, productResult out);
//int inferEngine(productParam &param, productFunc &func, std::vector<std::string> &imgPaths, productResult &out);

//extern "C" std::map<std::string, batchBoxesType> inferEngine(productParam &param, const InputData &data);
extern "C" int  inferEngine(Handle &engine, const InputData &data);
//std::map<std::string, batchBoxesType> inferEngine(const InputData &data);

//std::map<std::string, batchBoxesType> inferEngine(productParam &param, productFunc &func, std::string &imgPath);
//std::map<std::string, batchBoxesType> inferEngine(productParam &param, productFunc &func, std::vector<std::string> &imgPaths);
//std::map<std::string, batchBoxesType> inferEngine(productParam &param, productFunc &func, cv::Mat &mat);
//std::map<std::string, batchBoxesType> inferEngine(productParam &param, productFunc &func, std::vector<cv::Mat> &mats);
//std::map<std::string, batchBoxesType> inferEngine(productParam &param, productFunc &func, cv::cuda::GpuMat &gpuMat);
//std::map<std::string, batchBoxesType> inferEngine(productParam &param, productFunc &func, std::vector<cv::cuda::GpuMat> &gpuMats);

extern "C" int releaseEngine(Handle &engine);

int getResult(productParam &param, productResult &out);

/*
*   @brief                  获取人脸特征提取结果
*   @param res_num          输入人脸特征提取结果数量，inferFrame接口返回值
*   @param res_ptr          输出人脸特征提取结果结构体指针
*   @return                 成功返回0；失败返回对应错误码
*/
//int getResult(Handle engine, int res_num, FaceResult *res_ptr);

#endif //FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H