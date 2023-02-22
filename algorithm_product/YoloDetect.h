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
    int preProcess(cv::Mat &image, float *pinMemoryCurrentIn, parmBase base) override;
//    int preProcess(cv::Mat &image, float *pinMemoryCurrentIn, struct parmBase conf);

    // 图片后处理
    int postProcess(struct outputBase &result) override;
    int postProcess(std::vector<cv::Mat>,float *pinMemoryOut, parmBase conf) override;
    std::vector<std::vector<float>> decodeBox(int boxNum, int predictNum, float *pinOutput, int classNum, const float d2i[]);
    // 推理内存中图片
    int inferImages(const std::vector<cv::Mat> &inputImages, struct outputBase &result) override;
    // 推理gpu中图片
    int inferGpuImages(const std::vector<cv::cuda::GpuMat> &inputImages, struct outputBase &result) override;
};

#endif //TENSORRTMODELDEPLOYMENT_YOLODETECT_H
