//
// Created by Administrator on 2023/2/20.
//

#ifndef TENSORRTMODELDEPLOYMENT_YOLODETECT_H
#define TENSORRTMODELDEPLOYMENT_YOLODETECT_H

#include "../algorithm_factory/factory.h"
#include "../algorithm_factory/struct_data_type.h"
#include "../utils/box_utils.h"
#include "../utils/general.h"

class YoloDetect : public AlgorithmBase {
public:
    YoloDetect();
    ~YoloDetect() override;

    int initParam(void *param) override;
    // 图片预处理
    int preProcess(ParmBase &parm, cv::Mat &image, float *pinMemoryCurrentIn) override;
//    int preProcess(cv::Mat &image, float *pinMemoryCurrentIn, ParmBase conf);

    // 图片后处理
//    int postProcess(ResultBase &result) override;
    int postProcess(ParmBase &parm, std::vector<cv::Mat> &images,
                    float *pinMemoryOut, int singleOutputSize, ResultBase &result) override;
    std::vector<std::vector<float>> decodeBox(int boxNum, int predictNum, float *pinOutput, int classNum, std::vector<float> d2i);
    // 推理内存中图片
//    int inferImages(const std::vector<cv::Mat> &inputImages, ResultBase &result) override;
    // 推理gpu中图片
//    int inferGpuImages(const std::vector<cv::cuda::GpuMat> &inputImages, ResultBase &result) override;
};

#endif //TENSORRTMODELDEPLOYMENT_YOLODETECT_H
