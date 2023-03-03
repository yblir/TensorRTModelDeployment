//
// Created by Administrator on 2023/1/9.
//
#ifndef FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H
#define FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H
#include <tuple>

#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>
//#include "struct_data_type.h"
//#include "base_interface/ai_img_alg_base.h"
//#include "../algorithm_product/YoloFace.h"
#include "../algorithm_factory/factory.h"
#include "../algorithm_product/product.h"

#define checkRuntime(op) check_cuda_runtime((op),#op,__FILE__,__LINE__)

using Handle = void *;

bool check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line);

// 初始化过程中,各个模型都会用到的通用步骤
int initCommon(ParamBase &confSpecific, class AlgorithmBase *funcSpecific);

// 测试重构一下?
int inferEngine(productParam &parm, productFunc &func, std::vector<cv::Mat> &images, int &res_num);
int initEngine(productParam &parm, productFunc &func);
int inferEngine(productParam &parm, productFunc &func, std::vector<cv::Mat> &mats, productResult &out);
int inferEngine(productParam &parm, productFunc &func, std::vector<cv::cuda::GpuMat> &matVector, productResult out);

int releaseEngine(Handle engine);

int getResult(productParam &parm, productResult &out);

/*
*   @brief                  获取人脸特征提取结果
*   @param res_num          输入人脸特征提取结果数量，inferFrame接口返回值
*   @param res_ptr          输出人脸特征提取结果结构体指针
*   @return                 成功返回0；失败返回对应错误码
*/
//int getResult(Handle engine, int res_num, FaceResult *res_ptr);

#endif //FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H