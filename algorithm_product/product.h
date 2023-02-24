//
// Created by Administrator on 2023/2/15.
//

#ifndef TENSORRT_PRO_PRODUCT_H
#define TENSORRT_PRO_PRODUCT_H

#include "../algorithm_factory/struct_data_type.h"
#include "../algorithm_factory/factory.h"

// ----------------------------------------------------------
// 以下三个结构体,从参数,函数到结果,必须一一对应                    |
// ----------------------------------------------------------

// 接受从外部传入的配置参数,并传递给算法
struct productConfig {
    YoloFaceConfig yoloConfig;
    YoloDetectConfig detectConfig;
};

// 从so动态库接受实现的算法函数调用,所有算法实现都共同基类AlgorithmBase
struct productFunc {
    AlgorithmBase *yoloFace;
    AlgorithmBase *yoloDetect;
};

// 提取各个模型推理结果
struct productResult{
    YoloDetectResult detectResult;
};

#endif //TENSORRT_PRO_PRODUCT_H
