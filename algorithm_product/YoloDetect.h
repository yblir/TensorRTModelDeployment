//
// Created by Administrator on 2023/2/20.
//

#ifndef TENSORRTMODELDEPLOYMENT_YOLODETECT_H
#define TENSORRTMODELDEPLOYMENT_YOLODETECT_H

#include "../algorithm_factory/factory.h"
//#include "../algorithm_factory/struct_data_type.h"
#include "../utils/box_utils.h"
#include "../utils/general.h"
#include "product.h"

class YoloDetect : public AlgorithmBase {
public:
    YoloDetect();
    ~YoloDetect() override;

    int initParam(void *param) override;
    // 图片预处理
    int preProcess(ParamBase &parm, cv::Mat &image, float *pinMemoryCurrentIn) override;
    cv::Mat preProcess(ParamBase &param, cv::Mat &image);
    // 图片后处理
    int postProcess(ParamBase &parm, float *pinMemoryOut, int singleOutputSize,
                    int outputNums, std::vector<std::vector<std::vector<float>>> &result) override;


    std::vector<std::vector<float>> decodeBox(int predictNums, int predictLength,
                                              float *pinOutput, int classNum, float scoreThresh, std::vector<float> d2i);

    std::vector<std::vector<std::vector<float>>> getCurResult();
    // 推理内存中图片
//    int inferImages(const std::vector<cv::Mat> &inputImages, ResultBase &result) override;
    // 推理gpu中图片
//    int inferGpuImages(const std::vector<cv::cuda::GpuMat> &inputImages, ResultBase &result) override;
private:
    // 各具体算法定义自己的输出数据结构,并在getCurResult方法中返回.
    std::vector<std::vector<std::vector<float>>> m_curResult;
};

#endif //TENSORRTMODELDEPLOYMENT_YOLODETECT_H
