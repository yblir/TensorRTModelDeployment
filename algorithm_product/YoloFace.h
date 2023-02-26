//
// Created by Administrator on 2023/2/10.
//

#ifndef TENSORRT_PRO_YOLOFACE_H
#define TENSORRT_PRO_YOLOFACE_H


//#pragma once
#include "../algorithm_factory/factory.h"
#include "../algorithm_factory/struct_data_type.h"

class YoloFace : public AlgorithmBase {
public:
    YoloFace();
    ~YoloFace() override;

    int initParam(void *param) override;
    // 图片预处理
    int preProcess(ParmBase &parm, cv::Mat &image, float *pinMemoryCurrentIn) override;
    // 图片后处理
//    int postProcess(struct ResultBase &result) override;
    int postProcess(ParmBase &parm, float *pinMemoryOut, int singleOutputSize,
                    int outputNums, std::vector<std::vector<std::vector<float>>> &result) override;
//    // 推理内存中图片
//    int inferImages(const std::vector<cv::Mat> &inputImages, struct ResultBase &result) override;
//    // 推理gpu中图片
//    int inferGpuImages(const std::vector<cv::cuda::GpuMat> &inputImages, struct ResultBase &result) override;

//    YoloFaceConfig getArr() { return faceConfig; };
//    YoloFaceConfig *conf2;
//private:
//    YoloFaceConfig faceConfig;
};

//struct productConfig {
//    YoloFace *yoloFace;
//};

//extern "C" AlgorithmBase *MakeAlgorithm(void){
//    AlgorithmBase *curProduct=new YoloFace;
//    return curProduct;
//}

#endif //TENSORRT_PRO_YOLOFACE_H