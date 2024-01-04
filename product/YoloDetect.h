//
// Created by Administrator on 2023/2/20.
//

#ifndef TENSORRTMODELDEPLOYMENT_YOLODETECT_H
#define TENSORRTMODELDEPLOYMENT_YOLODETECT_H

#include "../base_infer/infer.h"
#include "../utils/box_utils.h"
#include "../utils/general.h"
#include "product.h"

class YoloDetect : public Infer {
public:
    YoloDetect();
    ~YoloDetect() override;
//弃用 引用形式
/*
    int preProcess(BaseParam &param, const cv::Mat &image, float *pinMemoryCurrentIn) override;
    int preProcess(BaseParam &param, const cv::Mat &image, float *pinMemoryCurrentIn, const int &index) override;
    int postProcess(BaseParam &param, float *pinMemoryCurrentOut, int singleOutputSize, int outputNums, batchBoxesType &result) override;
    int postProcess(BaseParam &param, float *pinMemoryCurrentOut, int singleOutputSize, const int &index, batchBoxesType &result) override;
    int preProcess(BaseParam &param, const pybind11::array &image, float *pinMemoryCurrentIn) override;
    int preProcess(BaseParam &param, const pybind11::array &image, float *pinMemoryCurrentIn, const int &index) override;
*/
    // 预处理

    int preProcess(BaseParam *param, const cv::Mat &image, float *pinMemoryCurrentIn) override;
    int preProcess(BaseParam *param, const cv::Mat &image, float *pinMemoryCurrentIn, const int &index) override;
    // 后处理
    int postProcess(BaseParam *param, float *pinMemoryCurrentOut, int singleOutputSize, int outputNums, batchBoxesType &result) override;
    int postProcess(BaseParam *param, float *pinMemoryCurrentOut, int singleOutputSize, std::map<int, imgBoxesType> &result, const int &index) override;

    int preProcess(BaseParam *param, const pybind11::array &image, float *pinMemoryCurrentIn) override;
    int preProcess(BaseParam *param, const pybind11::array &image, float *pinMemoryCurrentIn, const int &index) override;


    static std::vector<std::vector<float>> decodeBox(int predictNums, int predictLength,
                                                     float *pinOutput, int classNum, float scoreThresh, std::vector<float> d2i);

private:
    // 各具体算法定义自己的输出数据结构,并在getCurResult方法中返回.
    std::vector<std::vector<std::vector<float>>> m_curResult;
};

#endif //TENSORRTMODELDEPLOYMENT_YOLODETECT_H
